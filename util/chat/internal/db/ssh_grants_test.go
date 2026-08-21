package db

import (
	"path/filepath"
	"testing"
)

func TestSSHConversationGrantPersistsAndCascades(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ssh-grants.db")
	store, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.CreateSession("session-1", "SSH", "model", "low"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.CreateSSHHost(SSHHost{ID: "host-1", Alias: "main", Name: "Main", Hostname: "192.0.2.10", Port: 22, Username: "user", KeyID: "main", TimeoutSeconds: 60}); err != nil {
		t.Fatal(err)
	}
	if err := store.GrantSSHConversation("session-1", "host-1"); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	store, err = Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	has, err := store.HasSSHConversationGrant("session-1", "host-1")
	if err != nil || !has {
		t.Fatalf("grant after reopen=%v err=%v", has, err)
	}
	grants, err := store.SSHConversationGrants("session-1")
	if err != nil || len(grants) != 1 || grants[0].HostAlias != "main" {
		t.Fatalf("grants=%#v err=%v", grants, err)
	}
	if err := store.DeleteSession("session-1"); err != nil {
		t.Fatal(err)
	}
	has, err = store.HasSSHConversationGrant("session-1", "host-1")
	if err != nil || has {
		t.Fatalf("grant after session delete=%v err=%v", has, err)
	}
}
