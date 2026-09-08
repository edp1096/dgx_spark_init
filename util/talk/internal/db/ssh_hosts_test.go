package db

import (
	"path/filepath"
	"testing"
)

func TestSSHHostCRUD(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "ssh.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	host, err := store.CreateSSHHost(SSHHost{ID: "ssh-1", Alias: "dgx-main", Name: "DGX Spark", Hostname: "192.168.100.61", Port: 22, Username: "edp1096", KeyID: "dgx-main", TimeoutSeconds: 60})
	if err != nil {
		t.Fatal(err)
	}
	host.Name = "Main Spark"
	if _, err := store.UpdateSSHHost(host); err != nil {
		t.Fatal(err)
	}
	byAlias, err := store.SSHHostByAlias("dgx-main")
	if err != nil || byAlias.Name != "Main Spark" {
		t.Fatalf("host=%+v err=%v", byAlias, err)
	}
	if err := store.DeleteSSHHost(host.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := store.SSHHost(host.ID); err == nil {
		t.Fatal("deleted SSH host is still available")
	}
}
