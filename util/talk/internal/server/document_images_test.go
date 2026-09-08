package server

import (
	"encoding/json"
	"fmt"
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
