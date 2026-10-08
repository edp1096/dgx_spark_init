package main

import "testing"

func TestTargetAddressValidation(t *testing.T) {
	address, err := targetAddress("192.168.100.61", 22)
	if err != nil || address != "192.168.100.61:22" {
		t.Fatalf("address=%q err=%v", address, err)
	}
	for _, host := range []string{"", "bad host", "../host", "host\nnext"} {
		if _, err := targetAddress(host, 22); err == nil {
			t.Fatalf("expected invalid host %q", host)
		}
	}
}

func TestKeyIDValidation(t *testing.T) {
	for _, value := range []string{"dgx-main", "server_a.key", "key01"} {
		if !keyIDPattern.MatchString(value) {
			t.Fatalf("expected valid key id %q", value)
		}
	}
	for _, value := range []string{"", "../key", "/root/key", "key name"} {
		if keyIDPattern.MatchString(value) {
			t.Fatalf("expected invalid key id %q", value)
		}
	}
}
