package main

// The manifest is the commit point. Immutable objects are written and synced
// before it is replaced; readers and CLI writers share an OS file lock.
import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/knownhosts"
)

type keyEntry struct {
	storedKey
	Hash string `json:"hash"`
}
type keyManifest struct {
	Schema     int                 `json:"schema"`
	Repository string              `json:"repository"`
	Authority  string              `json:"authority"`
	Epoch      uint64              `json:"epoch"`
	Version    uint64              `json:"version"`
	Keys       map[string]keyEntry `json:"keys"`
	KnownHosts string              `json:"known_hosts_hash"`
}
type keyStoreStatus struct {
	Node     string      `json:"node"`
	Manifest keyManifest `json:"manifest"`
}
type keyArchive struct {
	Manifest keyManifest       `json:"manifest"`
	Objects  map[string][]byte `json:"objects"`
}
type keyStore struct {
	cfg  config
	node string
	m    keyManifest
}

func randomID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b[:])
}
func hashBytes(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func atomicPrivate(path string, data []byte) error {
	f, err := os.CreateTemp(filepath.Dir(path), ".write-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if err = f.Chmod(0600); err == nil {
		_, err = f.Write(data)
	}
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	if err = os.Rename(tmp, path); err != nil {
		return err
	}
	d, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}
func (a *api) withStore(fn func(*keyStore) error) error {
	if err := os.MkdirAll(a.cfg.KeyDir, 0700); err != nil {
		return err
	}
	f, err := os.OpenFile(filepath.Join(a.cfg.KeyDir, ".store.lock"), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return err
	}
	defer f.Close()
	if err = syscall.Flock(int(f.Fd()), syscall.LOCK_EX); err != nil {
		return err
	}
	defer syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
	s := &keyStore{cfg: a.cfg}
	if err = s.open(); err != nil {
		return err
	}
	return fn(s)
}
func (s *keyStore) objectPath(hash string) string {
	return filepath.Join(s.cfg.KeyDir, ".objects", hash)
}
func (s *keyStore) put(data []byte) (string, error) {
	h := hashBytes(data)
	return h, atomicPrivate(s.objectPath(h), data)
}
func (s *keyStore) open() error {
	if err := os.MkdirAll(filepath.Join(s.cfg.KeyDir, ".objects"), 0700); err != nil {
		return err
	}
	nodePath := filepath.Join(s.cfg.KeyDir, ".node-id")
	b, err := os.ReadFile(nodePath)
	if errors.Is(err, os.ErrNotExist) {
		b = []byte(randomID())
		err = atomicPrivate(nodePath, b)
	}
	if err != nil {
		return err
	}
	s.node = string(b)
	b, err = os.ReadFile(filepath.Join(s.cfg.KeyDir, ".manifest.json"))
	if err == nil {
		if err = json.Unmarshal(b, &s.m); err != nil {
			return err
		}
		return validateManifest(s.m)
	}
	if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	s.m = keyManifest{Schema: 1, Keys: map[string]keyEntry{}}
	entries, err := os.ReadDir(s.cfg.KeyDir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if e.IsDir() || !keyIDPattern.MatchString(e.Name()) {
			continue
		}
		p := filepath.Join(s.cfg.KeyDir, e.Name())
		meta, err := readStoredKey(p, e.Name())
		if err != nil {
			continue
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		h, err := s.put(b)
		if err != nil {
			return err
		}
		s.m.Keys[e.Name()] = keyEntry{meta, h}
	}
	trust, err := os.ReadFile(s.cfg.KnownHostsPath)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if len(trust) > 0 {
		s.m.KnownHosts, err = s.put(trust)
		if err != nil {
			return err
		}
	}
	if len(s.m.Keys) > 0 || s.m.KnownHosts != "" {
		s.claim()
	}
	if err = s.commit(); err != nil {
		return err
	}
	// Only migrated key files are removed. Unrecognized legacy files are retained.
	for id := range s.m.Keys {
		if err = os.Remove(filepath.Join(s.cfg.KeyDir, id)); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return nil
}
func (s *keyStore) claim() {
	s.m.Repository = randomID()
	s.m.Authority = s.node
	s.m.Epoch = 1
	s.m.Version = 1
}
func (s *keyStore) writable() error {
	if s.m.Repository == "" {
		s.claim()
	}
	if s.m.Authority != s.node {
		return errors.New("키 저장소가 복제본입니다. 현재 관리 호스트에서 수정하거나 관리 권한을 이전하세요")
	}
	return nil
}
func validateManifest(m keyManifest) error {
	if m.Schema != 1 || m.Keys == nil || len(m.Keys) > 256 {
		return errors.New("invalid key manifest")
	}
	if m.Repository == "" {
		if m.Authority != "" || m.Epoch != 0 || m.Version != 0 || len(m.Keys) > 0 || m.KnownHosts != "" {
			return errors.New("invalid empty repository")
		}
	} else if len(m.Repository) != 32 || len(m.Authority) != 32 || m.Epoch == 0 || m.Version == 0 {
		return errors.New("invalid repository authority")
	}
	for id, k := range m.Keys {
		if !keyIDPattern.MatchString(id) || k.ID != id || !validHash(k.Hash) {
			return errors.New("invalid key entry")
		}
	}
	if m.KnownHosts != "" && !validHash(m.KnownHosts) {
		return errors.New("invalid known_hosts hash")
	}
	return nil
}
func validHash(h string) bool { b, e := hex.DecodeString(h); return e == nil && len(b) == 32 }
func (s *keyStore) commit() error {
	b, e := json.MarshalIndent(s.m, "", "  ")
	if e != nil {
		return e
	}
	return atomicPrivate(filepath.Join(s.cfg.KeyDir, ".manifest.json"), b)
}
func (s *keyStore) cleanup() {
	used := map[string]bool{s.m.KnownHosts: true}
	for _, k := range s.m.Keys {
		used[k.Hash] = true
	}
	entries, _ := os.ReadDir(filepath.Join(s.cfg.KeyDir, ".objects"))
	for _, e := range entries {
		if !used[e.Name()] && !e.IsDir() {
			_ = os.Remove(s.objectPath(e.Name()))
		}
	}
}
func (s *keyStore) list() []storedKey {
	result := make([]storedKey, 0, len(s.m.Keys))
	for _, k := range s.m.Keys {
		result = append(result, k.storedKey)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].ID < result[j].ID })
	return result
}
func (s *keyStore) store(id string, data []byte, replace bool) (storedKey, error) {
	if !keyIDPattern.MatchString(id) || len(data) > maxPrivateKeyBytes {
		return storedKey{}, errors.New("invalid SSH key id or size")
	}
	signer, err := ssh.ParsePrivateKey(data)
	if err != nil {
		return storedKey{}, errors.New("invalid SSH private key")
	}
	if err = s.writable(); err != nil {
		return storedKey{}, err
	}
	_, exists := s.m.Keys[id]
	if !exists && len(s.m.Keys) >= 256 {
		return storedKey{}, errors.New("at most 256 SSH keys")
	}
	if exists && !replace {
		return storedKey{}, os.ErrExist
	}
	if replace && !exists {
		return storedKey{}, os.ErrNotExist
	}
	h, err := s.put(data)
	if err != nil {
		return storedKey{}, err
	}
	meta := keyMetadata(id, signer)
	s.m.Keys[id] = keyEntry{meta, h}
	s.m.Version++
	if err = s.commit(); err != nil {
		return storedKey{}, err
	}
	s.cleanup()
	return meta, nil
}
func (s *keyStore) remove(id string) error {
	if err := s.writable(); err != nil {
		return err
	}
	if _, ok := s.m.Keys[id]; !ok {
		return os.ErrNotExist
	}
	delete(s.m.Keys, id)
	s.m.Version++
	if err := s.commit(); err != nil {
		return err
	}
	s.cleanup()
	return nil
}
func (s *keyStore) archive() (keyArchive, error) {
	out := keyArchive{Manifest: s.m, Objects: map[string][]byte{}}
	hashes := []string{s.m.KnownHosts}
	for _, k := range s.m.Keys {
		hashes = append(hashes, k.Hash)
	}
	for _, h := range hashes {
		if h == "" {
			continue
		}
		b, err := os.ReadFile(s.objectPath(h))
		if err != nil {
			return out, err
		}
		if hashBytes(b) != h {
			return out, errors.New("key object hash mismatch")
		}
		out.Objects[h] = b
	}
	return out, nil
}
func (s *keyStore) apply(in keyArchive) error {
	if err := validateManifest(in.Manifest); err != nil {
		return err
	}
	m := in.Manifest
	if m.Repository == "" {
		return errors.New("cannot import an uninitialized repository")
	}
	if s.m.Repository != "" {
		if s.m.Repository != m.Repository {
			return errors.New("서로 다른 키 저장소입니다. 자동 병합하지 않습니다")
		}
		if m.Epoch < s.m.Epoch || (m.Epoch == s.m.Epoch && m.Version < s.m.Version) {
			return errors.New("stale key manifest")
		}
		if m.Epoch == s.m.Epoch && m.Authority != s.m.Authority {
			return errors.New("authority conflict")
		}
		if m.Epoch == s.m.Epoch && m.Version == s.m.Version {
			a, _ := json.Marshal(m)
			b, _ := json.Marshal(s.m)
			if string(a) != string(b) {
				return errors.New("key manifest conflict")
			}
			return nil
		}
		if s.m.Authority == s.node && m.Authority != s.node {
			return errors.New("현재 관리 호스트에서 권한 이전을 먼저 실행하세요")
		}
	}
	expected := map[string]bool{}
	for id, k := range m.Keys {
		data, ok := in.Objects[k.Hash]
		if !ok || len(data) > maxPrivateKeyBytes || hashBytes(data) != k.Hash {
			return errors.New("missing or corrupt private key")
		}
		signer, err := ssh.ParsePrivateKey(data)
		if err != nil || keyMetadata(id, signer) != k.storedKey {
			return errors.New("private key metadata mismatch")
		}
		expected[k.Hash] = true
	}
	if m.KnownHosts != "" {
		data, ok := in.Objects[m.KnownHosts]
		if !ok || len(data) > 1<<20 || hashBytes(data) != m.KnownHosts {
			return errors.New("invalid known_hosts object")
		}
		expected[m.KnownHosts] = true
	}
	for h := range expected {
		if _, err := s.put(in.Objects[h]); err != nil {
			return err
		}
	}
	s.m = m
	if err := s.commit(); err != nil {
		return err
	}
	s.cleanup()
	return nil
}
func (s *keyStore) handoff(target string) error {
	if len(target) != 32 || target == s.node {
		return errors.New("invalid target node")
	}
	if err := s.writable(); err != nil {
		return err
	}
	s.m.Authority = target
	s.m.Epoch++
	s.m.Version++
	return s.commit()
}
func (s *keyStore) trust(line string) error {
	if err := s.writable(); err != nil {
		return err
	}
	var b []byte
	var err error
	if s.m.KnownHosts != "" {
		b, err = os.ReadFile(s.objectPath(s.m.KnownHosts))
		if err != nil {
			return err
		}
	}
	if strings.Contains(string(b), line) {
		return nil
	}
	b = append(b, []byte(line+"\n")...)
	if len(b) > 1<<20 {
		return errors.New("known_hosts exceeds 1 MiB")
	}
	s.m.KnownHosts, err = s.put(b)
	if err != nil {
		return err
	}
	s.m.Version++
	return s.commit()
}

// Private archives are exposed ONLY through this local CLI (transported by SSH),
// never through an HTTP endpoint or application response/log.
func (a *api) keyStoreCLI(action string, input io.Reader, output io.Writer) error {
	return a.withStore(func(s *keyStore) error {
		var result any
		switch action {
		case "trust":
			var in struct {
				Host      string `json:"host"`
				Port      int    `json:"port"`
				PublicKey string `json:"public_key"`
			}
			if err := json.NewDecoder(io.LimitReader(input, 32768)).Decode(&in); err != nil {
				return err
			}
			address, err := targetAddress(in.Host, in.Port)
			if err != nil {
				return err
			}
			wanted, _, _, _, err := ssh.ParseAuthorizedKey([]byte(in.PublicKey))
			if err != nil {
				return errors.New("invalid host public key")
			}
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			observed, err := scanHostKey(ctx, address)
			if err != nil {
				return err
			}
			if !bytes.Equal(wanted.Marshal(), observed.Marshal()) {
				return errors.New("SSH host key changed before trust")
			}
			if err = s.trust(knownhosts.Line([]string{knownhosts.Normalize(address)}, observed)); err != nil {
				return err
			}
			result = map[string]string{"status": "trusted"}
		case "status":
			result = keyStoreStatus{s.node, s.m}
		case "list":
			result = s.list()
		case "export":
			var err error
			result, err = s.archive()
			if err != nil {
				return err
			}
		case "apply":
			var in keyArchive
			if err := json.NewDecoder(io.LimitReader(input, 48<<20)).Decode(&in); err != nil {
				return errors.New("invalid key archive")
			}
			if err := s.apply(in); err != nil {
				return err
			}
			result = keyStoreStatus{s.node, s.m}
		case "handoff":
			var in struct {
				Target string `json:"target"`
			}
			if err := json.NewDecoder(input).Decode(&in); err != nil {
				return err
			}
			if err := s.handoff(in.Target); err != nil {
				return err
			}
			result = keyStoreStatus{s.node, s.m}
		case "delete", "import", "replace", "generate":
			var in struct {
				ID   string `json:"id"`
				Data []byte `json:"data"`
			}
			if err := json.NewDecoder(io.LimitReader(input, 256<<10)).Decode(&in); err != nil {
				return errors.New("invalid key request")
			}
			if action == "delete" {
				if err := s.remove(in.ID); err != nil {
					return err
				}
				result = map[string]bool{"deleted": true}
			} else {
				var err error
				if action == "generate" {
					in.Data, err = generatePrivateKey()
					if err != nil {
						return err
					}
				}
				result, err = s.store(in.ID, in.Data, action == "replace")
				if err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("unknown key store command")
		}
		return json.NewEncoder(output).Encode(result)
	})
}
