package orchestrator

import (
	"strings"
	"testing"
)

func TestKeyAuthoritySelection(t *testing.T) {
	source := KeyReplica{Host: "head", Node: "a", Manifest: KeyManifest{Repository: "repo", Authority: "a", Epoch: 1, Version: 3}}
	replica := KeyReplica{Host: "worker", Node: "b", Manifest: KeyManifest{Repository: "repo", Authority: "a", Epoch: 1, Version: 2}}
	r := resolveKeyAuthority(KeyStoreReport{Replicas: []KeyReplica{replica, source}})
	if r.AuthorityHost != "head" || r.Error != "" {
		t.Fatalf("%+v", r)
	}
	source.Error = "offline"
	r = resolveKeyAuthority(KeyStoreReport{Replicas: []KeyReplica{source, replica}})
	if r.Error == "" || r.AuthorityHost != "" {
		t.Fatal("offline authority silently promoted replica")
	}
	source.Error = ""
	source.Manifest.Authority = "b"
	source.Manifest.Epoch = 2
	source.Manifest.Version = 4
	r = resolveKeyAuthority(KeyStoreReport{Replicas: []KeyReplica{source, replica}})
	if r.AuthorityHost != "worker" {
		t.Fatal("interrupted handoff cannot resume")
	}
	replica.Manifest.Repository = "foreign"
	r = resolveKeyAuthority(KeyStoreReport{Replicas: []KeyReplica{source, replica}})
	if r.Error == "" {
		t.Fatal("foreign repository merged")
	}
	replica = source
	replica.Node = "b"
	replica.Manifest.Authority = "b"
	source.Manifest.Authority = "a"
	r = resolveKeyAuthority(KeyStoreReport{Replicas: []KeyReplica{source, replica}})
	if !strings.Contains(r.Error, "충돌") {
		t.Fatal("equal version conflict accepted")
	}
}
func TestOfflineEmptyReplicaCannotInitializeNewAuthority(t *testing.T) {
	r := resolveKeyAuthority(KeyStoreReport{Replicas: []KeyReplica{{Host: "local", Node: "a"}, {Host: "worker", Error: "offline"}}})
	if r.Error == "" || r.AuthorityHost != "" {
		t.Fatal("initialized while another repository is unreachable")
	}
}

func TestKeyStorePeersRetainPhysicalHostAcrossMainChange(t *testing.T) {
	c, _ := NewController()
	original := map[string]Host{"local": {Address: "192.0.2.61", User: "alice", DataDir: "/keys/head"}, "worker": {Address: "192.0.2.60", User: "alice", DataDir: "/keys/worker"}}
	peers, err := c.PrepareKeyStorePeers([]string{"local", "worker"}, original)
	if err != nil {
		t.Fatal(err)
	}
	if peers["local"].Address != "192.0.2.61" || peers["worker"].DataDir != "/keys/worker" {
		t.Fatal("main change retargeted physical replicas")
	}
	peers["local"] = Host{}
	if original["local"].Address != "192.0.2.61" {
		t.Fatal("shared mutable peer map")
	}
	if !isLocalKeyHost("127.0.0.1") || isLocalKeyHost("192.0.2.61") {
		t.Fatal("local host detection")
	}
}
