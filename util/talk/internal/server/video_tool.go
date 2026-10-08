package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/llm"
)

func videoRuntime(cfg config.Config) (string, bool) {
	if cfg.Runtime.Catalog == nil {
		return "", false
	}
	b := cfg.Runtime.ActiveBundle
	if b == "" {
		b = cfg.Runtime.Bundle
	}
	x, ok := cfg.Runtime.Catalog.ResolveComponent(b, "qwim-mmh3")
	return strings.TrimRight(x.Endpoint, "/"), ok && x.ComposeAsset == "compose.qwim-mmh3.yaml"
}
func (s *Server) registerVideoTool(reg *completionToolRegistry, cfg config.Config, sink mediaAttachmentSink) {
	endpoint, ok := videoRuntime(cfg)
	if !ok || sink == nil {
		return
	}
	reg.prompts = append(reg.prompts, `Use video_generate when the user requests a new video. MiniMax H3 produces a 5.17-second 864x480 MP4 (124 frames at 24fps) with audio from an English prompt. Preserve requested scene, motion and sounds. Text-to-video only; do not claim image animation or arbitrary duration/resolution. It takes several minutes. Success is already stored and attached; never redownload or repeat a successful call. Describe the result briefly without claiming visual inspection.`)
	reg.register(llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "video_generate", Description: "Generate and attach a 5.17-second MiniMax H3 video with audio. Fixed 864x480/24fps. Text prompt only; generation may take minutes.", Parameters: json.RawMessage(`{"type":"object","properties":{"prompt":{"type":"string","minLength":1,"maxLength":8192},"seed":{"type":"integer","minimum":0,"maximum":9223372036854775807}},"required":["prompt"],"additionalProperties":false}`)}}, func(ctx context.Context, call llm.ToolCall, _ []llm.Message, emit eventEmitter) (registeredToolResult, error) {
		var args struct {
			Prompt string `json:"prompt"`
			Seed   *int64 `json:"seed,omitempty"`
		}
		dec := json.NewDecoder(strings.NewReader(call.Function.Arguments))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&args); err != nil {
			return registeredToolResult{}, err
		}
		if strings.TrimSpace(args.Prompt) == "" || len(args.Prompt) > 8192 || (args.Seed != nil && *args.Seed < 0) {
			return registeredToolResult{}, fmt.Errorf("invalid video prompt or seed")
		}
		generationStarted := time.Now()
		requestID, stopProgress := startGenerationProgress(ctx, endpoint, call.ID, "h3", emit)
		defer stopProgress()
		release, err := s.acquireWorkload(ctx, "qwim-mmh3")
		if err != nil {
			return registeredToolResult{}, err
		}
		var finalErr error
		defer func() { _ = release(finalErr) }()
		payload := map[string]any{"prompt": args.Prompt, "model": "minimax-h3-nvfp4"}
		if args.Seed != nil {
			payload["seed"] = *args.Seed
		}
		raw, _ := json.Marshal(payload)
		request, err := http.NewRequestWithContext(ctx, "POST", endpoint+"/v1/videos/generations", bytes.NewReader(raw))
		if err != nil {
			return registeredToolResult{}, err
		}
		request.Header.Set("Content-Type", "application/json")
		if requestID != "" {
			request.Header.Set("X-SparkTalk-Request-ID", requestID)
		}
		response, err := (&http.Client{Timeout: 30 * time.Minute}).Do(request)
		if err != nil {
			finalErr = err
			if ctx.Err() != nil {
				cancelCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
				defer cancel()
				req, _ := http.NewRequestWithContext(cancelCtx, "POST", endpoint+"/v1/runtime/cancel", nil)
				if reply, e := http.DefaultClient.Do(req); e == nil {
					reply.Body.Close()
				}
			}
			return registeredToolResult{}, err
		}
		defer response.Body.Close()
		if response.StatusCode != 200 {
			detail, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
			finalErr = fmt.Errorf("video service HTTP %d: %s", response.StatusCode, detail)
			return registeredToolResult{}, finalErr
		}
		contentType, _, _ := mime.ParseMediaType(response.Header.Get("Content-Type"))
		if contentType != "video/mp4" {
			finalErr = fmt.Errorf("video service did not return an MP4")
			return registeredToolResult{}, finalErr
		}
		stopProgress()
		if emit != nil {
			_ = emit("tool_output", map[string]any{"id": call.ID, "stream": "stdout", "delta": "대화에 저장 중…\n", "progress": generationProgress{Stage: "attaching", Kind: "h3", Elapsed: time.Since(generationStarted).Seconds()}})
		}
		attachment, err := s.media.SaveReader(response.Body, "MiniMax-H3.mp4", "video/mp4", s.media.Limits().MaxBytes())
		if err != nil {
			finalErr = err
			return registeredToolResult{}, err
		}
		if err = sink(attachment); err != nil {
			finalErr = err
			return registeredToolResult{}, err
		}
		if emit != nil {
			_ = emit("media_attached", attachment)
			_ = emit("tool_output", map[string]any{"id": call.ID, "stream": "stdout", "delta": "대화 저장 완료\n", "progress": generationProgress{Stage: "attached", Kind: "h3", Elapsed: time.Since(generationStarted).Seconds()}})
		}
		result, _ := json.Marshal(map[string]any{"status": "saved", "attachment": attachment, "width": 864, "height": 480, "frames": 124, "fps": 24, "duration_seconds": 124.0 / 24, "audio": true})
		return registeredToolResult{Result: string(result), Attachment: &attachment, AttachmentEmitted: true}, nil
	})
}
