package main

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"io"
	"net/http"
	"os"
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
	var keys []storedKey
	err := a.withStore(func(s *keyStore) error { keys = s.list(); return nil })
	if err != nil {
		writeError(w, 500, err.Error(), nil)
		return
	}
	writeJSON(w, 200, keys)
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
	data, err := generatePrivateKey()
	if err != nil {
		writeError(w, 500, err.Error(), nil)
		return
	}
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
	err := a.withStore(func(s *keyStore) error { return s.remove(keyID) })
	if err != nil {
		a.writeKeyStoreError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func generatePrivateKey() ([]byte, error) {
	_, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, err
	}
	encoded, err := x509.MarshalPKCS8PrivateKey(privateKey)
	if err != nil {
		return nil, err
	}
	return pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: encoded}), nil
}
func (a *api) storeKey(id string, data []byte) (storedKey, error) {
	// Preserve useful validation errors for uploads.
	_, err := ssh.ParsePrivateKey(data)
	if err != nil {
		var encrypted *ssh.PassphraseMissingError
		if errors.As(err, &encrypted) {
			return storedKey{}, errors.New("encrypted SSH private keys are not supported")
		}
		return storedKey{}, errors.New("invalid SSH private key")
	}
	var key storedKey
	err = a.withStore(func(s *keyStore) error { var err error; key, err = s.store(id, data, false); return err })
	return key, err
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
