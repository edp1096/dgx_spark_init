package media

import (
	"bytes"
	"testing"
)

func TestUploadZIP(t *testing.T) {
	d := testDocumentArchive(t, map[string]string{"src/main.py": "print('hello')"})
	s, e := New(t.TempDir() + "/chat.db")
	if e != nil {
		t.Fatal(e)
	}
	for _, mime := range []string{"application/zip", "application/x-zip-compressed", "application/octet-stream"} {
		a, e := s.SaveReader(bytes.NewReader(d), "project.zip", mime, MaxAttachmentBytes)
		if e != nil || a.MIME != "application/zip" {
			t.Fatalf("%s: %+v %v", mime, a, e)
		}
	}
	if _, e = s.SaveReader(bytes.NewReader([]byte("PKinvalid")), "bad.zip", "application/zip", MaxAttachmentBytes); e == nil {
		t.Fatal("invalid ZIP accepted")
	}
}
