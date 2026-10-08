package media

import (
	"archive/zip"
	"bytes"
	"fmt"
	"image"
	"image/png"
	"io"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sparktalk/internal/attachment"
	"sparktalk/internal/db"
	"strings"
	"testing"
)

type repeatByte byte

func (b repeatByte) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = byte(b)
	}
	return len(p), nil
}

func TestConfiguredLimitsStreamingAndExistingDownloads(t *testing.T) {
	s, err := New(filepath.Join(t.TempDir(), "talk.db"), attachment.Limits{MaxFileMB: 2, TypeLimitsMB: map[string]int{"document": 1}})
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int64{1 << 20, 1<<20 + 1} {
		item, err := s.SaveReader(io.LimitReader(repeatByte('a'), n), "data.txt", "", s.Limits().MaxBytes())
		if n == 1<<20 {
			if err != nil || item.Size != n {
				t.Fatalf("boundary failed: %v", err)
			}
		} else if err == nil {
			t.Fatal("override bypassed")
		}
	}
	// UTF-8 code points crossing 64 KiB boundaries must survive streaming validation.
	data := strings.Repeat("a", 65535) + "한글" + strings.Repeat("b", 3)
	item, err := s.SaveReader(strings.NewReader(data), "한글.txt", "", s.Limits().MaxBytes())
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{strings.Repeat("a", 65535) + "\xff", strings.Repeat("a", 65535) + "\x00", strings.Repeat("a", 65535) + "\xed\x95"} {
		if _, err = s.SaveReader(strings.NewReader(bad), "bad.txt", "", s.Limits().MaxBytes()); err == nil {
			t.Fatal("invalid text accepted")
		}
	}
	s.Configure(attachment.Limits{MaxFileMB: 1, TypeLimitsMB: map[string]int{"document": 2}})
	large, err := s.SaveReader(io.LimitReader(repeatByte('x'), 2<<20), "large.txt", "", s.Limits().MaxBytes())
	if err != nil {
		t.Fatal(err)
	}
	s.Configure(attachment.Limits{MaxFileMB: 1})
	if _, err = s.Validate([]db.Attachment{large}); err == nil {
		t.Fatal("new conversation bypassed lowered limit")
	}
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/", nil)
	r.Header.Set("Range", "bytes=0-9")
	s.Serve(w, r, large.ID, large.Name, large.MIME)
	if w.Code != 206 || w.Body.Len() != 10 {
		t.Fatal("existing download lost after setting lowered")
	}
	if _, err = s.Validate([]db.Attachment{item}); err != nil {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(s.dir)
	if len(entries) != 3 {
		t.Fatalf("failed saves leaked staged files: %d", len(entries))
	}
}

func TestDefault256MiBArchiveBoundary(t *testing.T) {
	if testing.Short() {
		t.Skip("large streaming boundary")
	}
	dir := t.TempDir()
	src, err := os.Create(filepath.Join(dir, "source.zip"))
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	// ZIP stored-entry overhead is independent of payload size below 4 GiB.
	var small bytes.Buffer
	z := zip.NewWriter(&small)
	f, _ := z.CreateHeader(&zip.FileHeader{Name: "data.bin", Method: zip.Store})
	_, _ = f.Write([]byte{0})
	_ = z.Close()
	limit := attachment.Default().MaxBytes()
	payload := limit - int64(small.Len()-1)
	z = zip.NewWriter(src)
	f, err = z.CreateHeader(&zip.FileHeader{Name: "data.bin", Method: zip.Store})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = io.Copy(f, io.LimitReader(repeatByte(0), payload)); err != nil {
		t.Fatal(err)
	}
	if err = z.Close(); err != nil {
		t.Fatal(err)
	}
	info, _ := src.Stat()
	if info.Size() != limit {
		t.Fatal(fmt.Sprintf("fixture size %d != %d", info.Size(), limit))
	}
	s, err := New(filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = src.Seek(0, 0)
	item, err := s.SaveReader(src, "boundary.zip", "application/zip", s.Limits().MaxBytes())
	if err != nil || item.Size != limit {
		t.Fatalf("256 MiB ZIP failed: %v", err)
	}
	_, _ = src.Seek(0, 0)
	if _, err = s.SaveReader(io.MultiReader(src, strings.NewReader("x")), "over.zip", "application/zip", s.Limits().MaxBytes()); err == nil {
		t.Fatal("256 MiB + 1 accepted")
	}
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/", nil)
	r.Header.Set("Range", "bytes=0-7")
	s.Serve(w, r, item.ID, item.Name, item.MIME)
	if w.Code != 206 || w.Body.Len() != 8 {
		t.Fatal("large ZIP cannot be downloaded with ranges")
	}
	entries, _ := os.ReadDir(s.dir)
	if len(entries) != 1 {
		t.Fatal("oversize save left staging file")
	}
}

func TestLargeImageKeepsOriginalAndUsesSmallModelInput(t *testing.T) {
	s, err := New(filepath.Join(t.TempDir(), "talk.db"))
	if err != nil {
		t.Fatal(err)
	}
	// A valid one-pixel PNG plus trailing bytes models a large downloadable
	// original without constructing a huge decoded image in this regression.
	var b bytes.Buffer
	if err = png.Encode(&b, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	raw := b.Bytes()
	item, err := s.SaveReader(io.MultiReader(bytes.NewReader(raw), io.LimitReader(repeatByte(0), MaxInlineBytes)), "large.png", "image/png", s.Limits().MaxBytes())
	if err != nil {
		t.Fatal(err)
	}
	url, err := s.DataURL(item)
	if err != nil || !strings.HasPrefix(url, "data:image/jpeg;base64,") || len(url) > 10000 {
		t.Fatalf("large model image: %v", err)
	}
	original, err := s.Open(item)
	if err != nil {
		t.Fatal(err)
	}
	defer original.Close()
	stat, _ := original.Stat()
	if stat.Size() != item.Size || stat.Size() <= MaxInlineBytes {
		t.Fatal("original attachment changed")
	}
}
