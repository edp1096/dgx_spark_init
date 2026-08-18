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
	item, err := store.SaveImage(header)
	if err != nil {
		t.Fatal(err)
	}
	if item.MIME != "image/png" || item.Name != "red.png" || item.URL == "" {
		t.Fatalf("unexpected attachment: %+v", item)
	}
	orphan, err := store.SaveImage(header)
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
	if _, _, err := store.read(orphan.ID, orphan.Name, orphan.MIME, false); err == nil {
		t.Fatal("unused image was not removed")
	}
	if _, _, err := store.read(item.ID, item.Name, item.MIME, false); err != nil {
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

func TestSupportedAudioAndVideoSignatures(t *testing.T) {
	tests := []struct {
		name     string
		declared string
		data     []byte
		want     string
	}{
		{"sound.mp3", "audio/mpeg", []byte("ID3sample"), "audio/mpeg"},
		{"sound.wav", "audio/wav", append(append([]byte("RIFF0000"), []byte("WAVE")...), []byte("sample")...), "audio/wav"},
		{"sound.ogg", "audio/ogg", []byte("OggSsample"), "audio/ogg"},
		{"movie.ogg", "video/ogg", []byte("OggSsample"), "video/ogg"},
		{"movie.avi", "video/x-msvideo", append(append([]byte("RIFF0000"), []byte("AVI ")...), []byte("sample")...), "video/x-msvideo"},
		{"movie.mov", "video/quicktime", append([]byte{0, 0, 0, 12}, []byte("ftypqt  ")...), "video/quicktime"},
		{"movie.mp4", "video/mp4", append([]byte{0, 0, 0, 12}, []byte("ftypisom")...), "video/mp4"},
		{"movie.wmv", "video/x-ms-wmv", append([]byte{0x30, 0x26, 0xb2, 0x75, 0x8e, 0x66, 0xcf, 0x11, 0xa6, 0xd9, 0x00, 0xaa, 0x00, 0x62, 0xce, 0x6c}, []byte("sample")...), "video/x-ms-wmv"},
		{"movie.webm", "video/webm", append([]byte{0x1a, 0x45, 0xdf, 0xa3}, []byte("sample")...), "video/webm"},
	}
	for _, test := range tests {
		t.Run(test.name+"/"+test.declared, func(t *testing.T) {
			got, err := classifyMedia(test.data, test.name, test.declared, false)
			if err != nil || got != test.want {
				t.Fatalf("classifyMedia() = %q, %v; want %q", got, err, test.want)
			}
		})
	}
}

func TestRejectsDisguisedMedia(t *testing.T) {
	if _, err := classifyMedia([]byte("not a video"), "fake.mp4", "video/mp4", false); err == nil {
		t.Fatal("disguised MP4 was accepted")
	}
}
