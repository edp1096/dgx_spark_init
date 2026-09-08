package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func testStore(t *testing.T) *api {
	t.Helper()
	d := t.TempDir()
	return newAPI(config{KeyDir: filepath.Join(d, "keys"), KnownHostsPath: filepath.Join(d, "known_hosts"), MaxConcurrency: 1})
}
func state(t *testing.T, a *api) keyStoreStatus {
	t.Helper()
	var st keyStoreStatus
	if err := a.withStore(func(s *keyStore) error { st = keyStoreStatus{s.node, s.m}; return nil }); err != nil {
		t.Fatal(err)
	}
	return st
}
func archive(t *testing.T, a *api) keyArchive {
	t.Helper()
	var ar keyArchive
	if err := a.withStore(func(s *keyStore) error { var err error; ar, err = s.archive(); return err }); err != nil {
		t.Fatal(err)
	}
	return ar
}
func applyArchive(a *api, ar keyArchive) error {
	return a.withStore(func(s *keyStore) error { return s.apply(ar) })
}
func addKey(t *testing.T, a *api, id string) {
	t.Helper()
	key, err := generatePrivateKey()
	if err != nil {
		t.Fatal(err)
	}
	if _, err = a.storeKey(id, key); err != nil {
		t.Fatal(err)
	}
}
func TestKeyStoreDeletionAndAuthorityHandoff(t *testing.T) {
	a, b := testStore(t), testStore(t)
	addKey(t, a, "one")
	initial := archive(t, a)
	if err := applyArchive(b, initial); err != nil {
		t.Fatal(err)
	}
	if err := b.withStore(func(s *keyStore) error { return s.remove("one") }); err == nil {
		t.Fatal("replica accepted mutation")
	}
	if err := a.withStore(func(s *keyStore) error { return s.remove("one") }); err != nil {
		t.Fatal(err)
	}
	if err := applyArchive(b, archive(t, a)); err != nil {
		t.Fatal(err)
	}
	if len(state(t, b).Manifest.Keys) != 0 {
		t.Fatal("deletion was not replicated")
	}
	if err := applyArchive(b, initial); err == nil {
		t.Fatal("stale archive resurrected deleted key")
	}
	target := state(t, b).Node
	if err := a.withStore(func(s *keyStore) error { return s.handoff(target) }); err != nil {
		t.Fatal(err)
	}
	data, _ := generatePrivateKey()
	if _, err := a.storeKey("fenced", data); err == nil {
		t.Fatal("old owner still writable before transfer delivery")
	}
	// Interrupted delivery can be retried; only destination can write afterwards.
	next := archive(t, a)
	if err := applyArchive(b, next); err != nil {
		t.Fatal(err)
	}
	if err := applyArchive(b, next); err != nil {
		t.Fatal(err)
	}
	addKey(t, b, "new")
	if err := applyArchive(a, archive(t, b)); err != nil {
		t.Fatal(err)
	}
	if _, ok := state(t, a).Manifest.Keys["new"]; !ok {
		t.Fatal("reverse replication failed")
	}
}
func TestKeyStoreRejectsCorruptionAndForeignRepository(t *testing.T) {
	a, b := testStore(t), testStore(t)
	addKey(t, a, "one")
	ar := archive(t, a)
	for hash := range ar.Objects {
		ar.Objects[hash] = []byte("broken")
	}
	if err := applyArchive(b, ar); err == nil {
		t.Fatal("corrupt archive accepted")
	}
	if state(t, b).Manifest.Repository != "" {
		t.Fatal("partial archive committed")
	}
	addKey(t, b, "other")
	if err := applyArchive(b, archive(t, a)); err == nil {
		t.Fatal("foreign repository overwritten")
	}
}
func TestKeyStoreMigrationAndReplacement(t *testing.T) {
	a := testStore(t)
	if err := os.MkdirAll(a.cfg.KeyDir, 0700); err != nil {
		t.Fatal(err)
	}
	data, _ := generatePrivateKey()
	legacy := filepath.Join(a.cfg.KeyDir, "legacy")
	if err := os.WriteFile(legacy, data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(a.cfg.KnownHostsPath, []byte("example ssh-ed25519 example\n"), 0600); err != nil {
		t.Fatal(err)
	}
	st := state(t, a)
	if len(st.Manifest.Keys) != 1 || st.Manifest.KnownHosts == "" {
		t.Fatal("legacy state missing")
	}
	if _, err := os.Stat(legacy); !os.IsNotExist(err) {
		t.Fatal("legacy private copy retained")
	}
	path := filepath.Join(a.cfg.KeyDir, ".objects", st.Manifest.Keys["legacy"].Hash)
	replacement, _ := generatePrivateKey()
	if err := a.withStore(func(s *keyStore) error { _, err := s.store("legacy", replacement, true); return err }); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("replaced private object retained")
	}
	b := testStore(t)
	if err := applyArchive(b, archive(t, a)); err != nil {
		t.Fatal(err)
	}
	if state(t, b).Manifest.Keys["legacy"].Fingerprint == st.Manifest.Keys["legacy"].Fingerprint {
		t.Fatal("replacement failed")
	}
}
func TestKeyStoreConcurrentWritersAndManifestOnlyStatus(t *testing.T) {
	a := testStore(t)
	var wg sync.WaitGroup
	for _, id := range []string{"a", "b", "c", "d"} {
		wg.Add(1)
		go func(id string) {
			defer wg.Done()
			data, _ := generatePrivateKey()
			if _, err := a.storeKey(id, data); err != nil {
				t.Error(err)
			}
		}(id)
	}
	wg.Wait()
	st := state(t, a)
	if len(st.Manifest.Keys) != 4 {
		t.Fatal("lost concurrent write")
	}
	// Status reads the manifest, not every private object. Export detects damage.
	for _, k := range st.Manifest.Keys {
		_ = os.Remove(filepath.Join(a.cfg.KeyDir, ".objects", k.Hash))
		break
	}
	var out bytes.Buffer
	if err := a.keyStoreCLI("status", bytes.NewReader(nil), &out); err != nil {
		t.Fatal(err)
	}
	var status keyStoreStatus
	if err := json.Unmarshal(out.Bytes(), &status); err != nil {
		t.Fatal(err)
	}
	if err := a.withStore(func(s *keyStore) error { _, err := s.archive(); return err }); err == nil {
		t.Fatal("missing object not detected")
	}
}
