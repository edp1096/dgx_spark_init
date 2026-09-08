package server

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"sparktalk/internal/db"
)

func (s *Server) videoFrameSheet(ctx context.Context, item db.Attachment, endpoint string) (string, string, error) {
	file, err := s.media.Open(item)
	if err != nil {
		return "", "", err
	}
	defer file.Close()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(endpoint, "/")+"/v1/video/frames", file)
	if err != nil {
		return "", "", err
	}
	req.Header.Set("Content-Type", item.MIME)
	resp, err := (&http.Client{Timeout: 5 * time.Minute}).Do(req)
	if err != nil {
		return "", "", fmt.Errorf("영상 프레임 추출: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("영상 프레임 추출: HTTP %d (Extra Media 업데이트와 연결을 확인하세요)", resp.StatusCode)
	}
	const limit = 8 << 20
	data, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return "", "", err
	}
	if len(data) > limit || http.DetectContentType(data) != "image/jpeg" {
		return "", "", fmt.Errorf("invalid video frame image")
	}
	return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(data), resp.Header.Get("X-Video-Duration"), nil
}
