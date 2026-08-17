package media

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"mime/multipart"
	"net/http/httptest"
	"strings"
	"testing"

	"sparktalk/internal/db"
)

func TestImageLifecycle(t *testing.T) {
	store, err := New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, _ := writer.CreateFormFile("image", "red.png")
	picture := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			picture.Set(x, y, color.RGBA{R: 255, A: 255})
		}
	}
	if err := png.Encode(part, picture); err != nil {
		t.Fatal(err)
	}
	_ = writer.Close()

	request := httptest.NewRequest("POST", "/api/images", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	if err := request.ParseMultipartForm(1 << 20); err != nil {
		t.Fatal(err)
	}
	file, header, err := request.FormFile("image")
	if err != nil {
		t.Fatal(err)
	}
	_ = file.Close()
	item, err := store.Save(header)
	if err != nil {
		t.Fatal(err)
	}
	if item.MIME != "image/png" || item.Name != "red.png" || item.URL == "" {
		t.Fatalf("unexpected attachment: %+v", item)
	}
	orphan, err := store.Save(header)
	if err != nil {
		t.Fatal(err)
	}
	usage, err := store.Usage(map[string]struct{}{item.ID: {}}, nil)
	if err != nil || usage.Files != 2 || usage.UnusedFiles != 1 {
		t.Fatalf("unexpected media usage: usage=%+v err=%v", usage, err)
	}
	removed, err := store.Cleanup(map[string]struct{}{item.ID: {}}, nil)
	if err != nil || removed.Files != 1 {
		t.Fatalf("cleanup: removed=%+v err=%v", removed, err)
	}
	if _, _, err := store.read(orphan.ID); err == nil {
		t.Fatal("unused image was not removed")
	}
	if _, _, err := store.read(item.ID); err != nil {
		t.Fatalf("referenced image was removed: %v", err)
	}
	validated, err := store.Validate([]db.Attachment{item})
	if err != nil || len(validated) != 1 {
		t.Fatalf("validate image: items=%+v err=%v", validated, err)
	}
	dataURL, err := store.DataURL(item)
	if err != nil || !strings.HasPrefix(dataURL, "data:image/png;base64,") {
		t.Fatalf("unexpected data URL: prefix=%q err=%v", dataURL[:min(len(dataURL), 32)], err)
	}
}
