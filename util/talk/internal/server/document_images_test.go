package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"sparktalk/internal/db"
	"strings"
	"testing"
)

func TestDocumentImagesRequireConversationAttachment(t *testing.T) {
	s, image := testImageServer(t)
	good := fmt.Sprintf(`{"format":"hwp","title":"문서","sections":[{"paragraphs":["본문"],"images":[{"image_id":%q,"caption":"첨부 그림"}]}]}`, image.ID)
	payload, err := s.documentImagePayload("session", good)
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Sections []struct {
			Images []struct {
				Data    string  `json:"data"`
				Width   int     `json:"width_px"`
				WidthCM float64 `json:"width_cm"`
			}
		}
	}
	if err = json.Unmarshal(payload, &doc); err != nil {
		t.Fatal(err)
	}
	if len(doc.Sections[0].Images) != 1 || doc.Sections[0].Images[0].Data == "" || doc.Sections[0].Images[0].Width != 1 || doc.Sections[0].Images[0].WidthCM != 12 {
		t.Fatalf("Image was not resolved: %s", payload)
	}
	for _, bad := range []string{`{"sections":[{"images":[{"image_id":"not-in-session"}]}]}`, `{"sections":[{"images":[{"data":"iVBOR","image_id":"anything"}]}]}`, fmt.Sprintf(`{"sections":[{"images":[{"image_id":%q,"width_cm":99}]}]}`, image.ID)} {
		if _, err := s.documentImagePayload("session", bad); err == nil {
			t.Fatalf("Invalid reference accepted: %s", bad)
		}
	}
	if _, err := s.documentImagePayload("", good); err == nil {
		t.Fatal("Image used without a conversation")
	}
}

func TestOrderedDocumentImagesInAllLayouts(t *testing.T) {
	s, image := testImageServer(t)
	for _, template := range []string{
		`{"format":"pptx","title":"x","slides":[{"title":"x","blocks":[{"type":"columns","columns":[[{"type":"image","image":{"image_id":%q}}],[{"type":"paragraph","text":"body"}]]}]}]}`,
		`{"format":"xlsx","title":"x","sheets":[{"name":"x","images":[{"cell":"A5","image":{"image_id":%q}}]}]}`,
		`{"format":"docx","title":"x","sections":[{"blocks":[{"type":"table","rows":[[{"blocks":[{"type":"image","image":{"image_id":%q}}]}]]}]}]}`,
	} {
		payload, err := s.documentImagePayload("session", fmt.Sprintf(template, image.ID))
		if err != nil {
			t.Fatal(err)
		}
		var tree any
		if err = json.Unmarshal(payload, &tree); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(payload), `"width_px":1`) || strings.Contains(string(payload), `"image_id"`) {
			t.Fatalf("Image not resolved: %s", payload)
		}
	}
	for _, args := range []string{`{"slides":[{"blocks":[{"type":"image","image":{"data":"AAAA"}}]}]}`, `{"slides":[{"blocks":[{"type":"media","media":{"data":"AAAA","mime":"video/mp4"}}]}]}`} {
		if _, err := s.documentImagePayload("session", args); err == nil {
			t.Fatal("Raw attachment bytes were accepted")
		}
	}
}

func TestPPTXMediaResolvesOnlySessionFiles(t *testing.T) {
	s, _ := testImageServer(t)
	data := []byte("RIFF0000WAVEfmt 1234567890")
	a, err := s.media.SaveReader(bytes.NewReader(data), "test.wav", "audio/wav", 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = s.db.AddMessage("session", "user", "音声", "", nil, []db.Attachment{a}); err != nil {
		t.Fatal(err)
	}
	args := fmt.Sprintf(`{"format":"pptx","title":"media","slides":[{"title":"media","blocks":[{"type":"media","media":{"media_id":%q}}]}]}`, a.ID)
	result, err := s.documentImagePayload("session", args)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(result), base64.StdEncoding.EncodeToString(data)) {
		t.Fatalf("Missing encoded media: %s", result)
	}
	if !strings.Contains(documentMediaCatalog(s, "session"), a.ID) {
		t.Fatal("Media ID not exposed to model")
	}
	if _, err = s.documentImagePayload("other-session", args); err == nil {
		t.Fatal("Cross-session media accepted")
	}
}
