package media

import (
	"bytes"
	"path/filepath"
	"testing"
)

func TestStreamingHWPHeaderBeyondInitialBytes(t *testing.T) {
	store, e := New(filepath.Join(t.TempDir(), "db"))
	if e != nil {
		t.Fatal(e)
	}
	data := make([]byte, 70000)
	copy(data, []byte{0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1})
	copy(data[65530:], []byte("HWP Document File\x00"))
	item, e := store.SaveReader(bytes.NewReader(data), "document.hwp", "application/x-hwp", MaxAttachmentBytes)
	if e != nil || item.MIME != "application/x-hwp" {
		t.Fatal("valid HWP rejected beyond header boundary", e)
	}
	bad := append([]byte(nil), data...)
	copy(bad[65530:], bytes.Repeat([]byte{0}, 20))
	if _, e = store.SaveReader(bytes.NewReader(bad), "fake.hwp", "application/x-hwp", MaxAttachmentBytes); e == nil {
		t.Fatal("OLE file without HWP signature accepted")
	}
}
