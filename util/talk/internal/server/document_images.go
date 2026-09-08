package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"io"
	"math"

	"golang.org/x/image/draw"
	"sparktalk/internal/db"
	"sparktalk/internal/media"
)

// Only conversation attachments are resolved; tool arguments cannot supply paths,
// URLs or arbitrary embedded image bytes to the document service.
func (s *Server) documentImagePayload(sessionID string, arguments string) ([]byte, error) {
	var root map[string]json.RawMessage
	if err := json.Unmarshal([]byte(arguments), &root); err != nil {
		return nil, err
	}
	var sections []map[string]json.RawMessage
	if raw, ok := root["sections"]; ok {
		if err := json.Unmarshal(raw, &sections); err != nil {
			return nil, err
		}
	}
	var available map[string]db.Attachment
	count, total := 0, 0
	for _, section := range sections {
		raw, ok := section["images"]
		if !ok {
			continue
		}
		var refs []struct {
			ImageID string  `json:"image_id"`
			WidthCM float64 `json:"width_cm"`
			Caption string  `json:"caption"`
		}
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&refs); err != nil {
			return nil, fmt.Errorf("잘못된 문서 이미지 참조: %w", err)
		}
		if len(refs) == 0 {
			continue
		}
		if sessionID == "" {
			return nil, fmt.Errorf("문서 이미지는 현재 대화에 첨부된 이미지만 사용할 수 있습니다")
		}
		if available == nil {
			var err error
			available, err = s.sessionImageAttachments(sessionID)
			if err != nil {
				return nil, err
			}
		}
		images := make([]map[string]any, 0, len(refs))
		for _, ref := range refs {
			count++
			if count > 6 {
				return nil, fmt.Errorf("문서 이미지 최대 6개를 초과했습니다")
			}
			item, ok := available[ref.ImageID]
			if !ok {
				return nil, fmt.Errorf("현재 대화에서 사용할 수 없는 이미지입니다: %s", ref.ImageID)
			}
			if ref.WidthCM == 0 {
				ref.WidthCM = 12
			}
			if math.IsNaN(ref.WidthCM) || math.IsInf(ref.WidthCM, 0) || ref.WidthCM < 1 || ref.WidthCM > 16 {
				return nil, fmt.Errorf("이미지 너비는 1–16 cm 범위여야 합니다")
			}
			file, err := s.media.Open(item)
			if err != nil {
				return nil, err
			}
			data, err := io.ReadAll(io.LimitReader(file, media.MaxImageBytes+1))
			file.Close()
			if err != nil {
				return nil, err
			}
			if len(data) > media.MaxImageBytes {
				return nil, fmt.Errorf("이미지 크기 제한을 초과했습니다")
			}
			cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
			if err != nil || cfg.Width < 1 || cfg.Height < 1 || int64(cfg.Width)*int64(cfg.Height) > 40_000_000 {
				return nil, fmt.Errorf("잘못된 이미지 크기입니다")
			}
			decoded, _, err := image.Decode(bytes.NewReader(data))
			if err != nil {
				return nil, err
			}
			w, h := cfg.Width, cfg.Height
			if max(w, h) > 1024 {
				scale := 1024.0 / float64(max(w, h))
				w = max(1, int(float64(w)*scale))
				h = max(1, int(float64(h)*scale))
				resized := image.NewNRGBA(image.Rect(0, 0, w, h))
				draw.ApproxBiLinear.Scale(resized, resized.Bounds(), decoded, decoded.Bounds(), draw.Src, nil)
				decoded = resized
			}
			var encoded bytes.Buffer
			if err := png.Encode(&encoded, decoded); err != nil {
				return nil, err
			}
			total += encoded.Len()
			if total > 8<<20 {
				return nil, fmt.Errorf("문서 이미지 합계가 8 MiB를 초과했습니다")
			}
			images = append(images, map[string]any{"data": base64.StdEncoding.EncodeToString(encoded.Bytes()), "width_px": w, "height_px": h, "width_cm": ref.WidthCM, "caption": ref.Caption})
		}
		normalized, err := json.Marshal(images)
		if err != nil {
			return nil, err
		}
		section["images"] = normalized
	}
	if sections != nil {
		normalized, err := json.Marshal(sections)
		if err != nil {
			return nil, err
		}
		root["sections"] = normalized
	}
	return json.Marshal(root)
}
