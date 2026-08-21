package main

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/crypto/ssh"
)

const maxPrivateKeyBytes = 128 * 1024

type storedKey struct {
	ID          string `json:"id"`
	Type        string `json:"type"`
	Fingerprint string `json:"fingerprint"`
	PublicKey   string `json:"public_key"`
}

func (a *api) listKeys(w http.ResponseWriter, _ *http.Request) {
	a.keysMu.Lock()
	defer a.keysMu.Unlock()
	entries, err := os.ReadDir(a.cfg.KeyDir)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "read SSH key directory: "+err.Error(), nil)
		return
	}
	keys := make([]storedKey, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !keyIDPattern.MatchString(entry.Name()) {
			continue
		}
		key, err := readStoredKey(filepath.Join(a.cfg.KeyDir, entry.Name()), entry.Name())
		if err == nil {
			keys = append(keys, key)
		}
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i].ID < keys[j].ID })
	writeJSON(w, http.StatusOK, keys)
}

func (a *api) generateKey(w http.ResponseWriter, r *http.Request) {
	var req struct {
		KeyID string `json:"key_id"`
	}
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), nil)
		return
	}
	req.KeyID = strings.TrimSpace(req.KeyID)
	if !keyIDPattern.MatchString(req.KeyID) {
		writeError(w, http.StatusBadRequest, "invalid SSH key id", nil)
		return
	}
	_, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "generate SSH key: "+err.Error(), nil)
		return
	}
	encoded, err := x509.MarshalPKCS8PrivateKey(privateKey)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "encode SSH key: "+err.Error(), nil)
		return
	}
	data := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: encoded})
	key, err := a.storeKey(req.KeyID, data)
	if err != nil {
		a.writeKeyStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, key)
}

func (a *api) importKey(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxPrivateKeyBytes+64*1024)
	if err := r.ParseMultipartForm(maxPrivateKeyBytes + 64*1024); err != nil {
		writeError(w, http.StatusBadRequest, "invalid or oversized SSH key upload", nil)
		return
	}
	keyID := strings.TrimSpace(r.FormValue("key_id"))
	if !keyIDPattern.MatchString(keyID) {
		writeError(w, http.StatusBadRequest, "invalid SSH key id", nil)
		return
	}
	file, _, err := r.FormFile("key")
	if err != nil {
		writeError(w, http.StatusBadRequest, "SSH private key file is required", nil)
		return
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxPrivateKeyBytes+1))
	if err != nil || len(data) == 0 || len(data) > maxPrivateKeyBytes {
		writeError(w, http.StatusBadRequest, "SSH private key must be between 1 byte and 128 KiB", nil)
		return
	}
	key, err := a.storeKey(keyID, data)
	if err != nil {
		a.writeKeyStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, key)
}

func (a *api) deleteKey(w http.ResponseWriter, r *http.Request) {
	keyID := strings.TrimSpace(r.PathValue("id"))
	if !keyIDPattern.MatchString(keyID) {
		writeError(w, http.StatusBadRequest, "invalid SSH key id", nil)
		return
	}
	a.keysMu.Lock()
	defer a.keysMu.Unlock()
	path := filepath.Join(a.cfg.KeyDir, keyID)
	if err := os.Remove(path); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			writeError(w, http.StatusNotFound, "SSH key not found", nil)
		} else {
			writeError(w, http.StatusInternalServerError, "delete SSH key: "+err.Error(), nil)
		}
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (a *api) storeKey(keyID string, data []byte) (storedKey, error) {
	signer, err := ssh.ParsePrivateKey(data)
	if err != nil {
		var passphraseErr *ssh.PassphraseMissingError
		if errors.As(err, &passphraseErr) {
			return storedKey{}, errors.New("encrypted SSH private keys are not supported")
		}
		return storedKey{}, errors.New("invalid SSH private key")
	}
	a.keysMu.Lock()
	defer a.keysMu.Unlock()
	path := filepath.Join(a.cfg.KeyDir, keyID)
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return storedKey{}, err
	}
	written := false
	defer func() {
		_ = file.Close()
		if !written {
			_ = os.Remove(path)
		}
	}()
	if _, err := file.Write(data); err != nil {
		return storedKey{}, fmt.Errorf("write SSH key: %w", err)
	}
	if err := file.Sync(); err != nil {
		return storedKey{}, fmt.Errorf("sync SSH key: %w", err)
	}
	if err := file.Close(); err != nil {
		return storedKey{}, fmt.Errorf("close SSH key: %w", err)
	}
	written = true
	return keyMetadata(keyID, signer), nil
}

func readStoredKey(path, keyID string) (storedKey, error) {
	info, err := os.Stat(path)
	if err != nil || info.IsDir() || info.Mode().Perm()&0o077 != 0 || info.Size() > maxPrivateKeyBytes {
		return storedKey{}, errors.New("unsafe SSH key file")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return storedKey{}, err
	}
	signer, err := ssh.ParsePrivateKey(data)
	if err != nil {
		return storedKey{}, err
	}
	return keyMetadata(keyID, signer), nil
}

func keyMetadata(keyID string, signer ssh.Signer) storedKey {
	publicKey := signer.PublicKey()
	return storedKey{
		ID: keyID, Type: publicKey.Type(), Fingerprint: ssh.FingerprintSHA256(publicKey),
		PublicKey: strings.TrimSpace(string(ssh.MarshalAuthorizedKey(publicKey))),
	}
}

func (a *api) writeKeyStoreError(w http.ResponseWriter, err error) {
	if errors.Is(err, os.ErrExist) {
		writeError(w, http.StatusConflict, "SSH key id already exists", nil)
		return
	}
	message := err.Error()
	if message == "invalid SSH private key" || message == "encrypted SSH private keys are not supported" {
		writeError(w, http.StatusBadRequest, message, nil)
		return
	}
	writeError(w, http.StatusInternalServerError, message, nil)
}
