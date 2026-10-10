package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"mime"
	"net/http"
	"strconv"
	"strings"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
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

const videoToolSystemPrompt = `Use video_generate when the user requests a new video. MiniMax H3 generates a 5.17-second MP4 with audio. Write a detailed English prompt preserving the requested subject, motion and sounds. Image-to-video is supported: when the user asks to animate an image or use it as frame 0, supply its actual conversation attachment ID as first_frame_image_id. Optional last_frame_image_id anchors the final frame. These images are encoded into the video model, not merely described in a text prompt. Do not substitute text-only generation for a requested image-conditioned video. Use only listed image attachment IDs, never a video ID. Inspect the selected image to write a suitable motion prompt. Image-conditioned output automatically follows the source aspect ratio; use the returned dimensions. Text-only output is 864x480/24fps. Arbitrary duration and reference video/audio inputs are unavailable. Generation takes several minutes. Success is already stored and attached; never redownload or repeat a successful call. Describe the result briefly without claiming visual inspection of the generated video.`

func (s *Server) registerVideoTool(ctx context.Context, reg *completionToolRegistry, cfg config.Config, sink mediaAttachmentSink) {
	endpoint, ok := videoRuntime(cfg)
	if !ok || sink == nil {
		return
	}
	reg.prompts = append(reg.prompts, videoToolSystemPrompt, imageAttachmentCatalogForContext(ctx, s, reg.sessionID))
	reg.register(llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "video_generate", Description: "Generate and attach a 5.17-second MiniMax H3 video with audio at 24fps. Supports text-to-video and image-to-video using actual conversation image IDs as first/last frame anchors. Image-conditioned size follows the source aspect ratio. Generation may take minutes.", Parameters: json.RawMessage(`{"type":"object","properties":{"prompt":{"type":"string","minLength":1,"maxLength":8192},"seed":{"type":"integer","minimum":0,"maximum":9223372036854775807},"first_frame_image_id":{"type":"string","minLength":1,"description":"Actual conversation image attachment ID to encode as the first frame (frame 0). Required when animating an input image."},"last_frame_image_id":{"type":"string","minLength":1,"description":"Optional conversation image attachment ID to encode as the final frame (frame 123)."}},"required":["prompt"],"additionalProperties":false}`)}}, func(ctx context.Context, call llm.ToolCall, _ []llm.Message, emit eventEmitter) (registeredToolResult, error) {
		var args struct {
			Prompt            string  `json:"prompt"`
			Seed              *int64  `json:"seed,omitempty"`
			FirstFrameImageID *string `json:"first_frame_image_id,omitempty"`
			LastFrameImageID  *string `json:"last_frame_image_id,omitempty"`
		}
		dec := json.NewDecoder(strings.NewReader(call.Function.Arguments))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&args); err != nil {
			return registeredToolResult{}, err
		}
		if strings.TrimSpace(args.Prompt) == "" || len(args.Prompt) > 8192 || (args.Seed != nil && *args.Seed < 0) {
			return registeredToolResult{}, fmt.Errorf("invalid video prompt or seed")
		}
		payload := map[string]any{"prompt": args.Prompt, "model": "minimax-h3-nvfp4"}
		if args.Seed != nil {
			payload["seed"] = *args.Seed
		}
		for _, frame := range []struct {
			key string
			id  *string
		}{{"first_frame", args.FirstFrameImageID}, {"last_frame", args.LastFrameImageID}} {
			if frame.id == nil {
				continue
			}
			data, err := s.videoKeyframeData(ctx, reg.sessionID, *frame.id)
			if err != nil {
				return registeredToolResult{}, err
			}
			payload[frame.key] = data
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
		width, height := 864, 480
		mode := "t2v"
		if args.FirstFrameImageID != nil || args.LastFrameImageID != nil {
			mode = "i2v"
			var widthErr, heightErr error
			width, widthErr = strconv.Atoi(response.Header.Get("X-Video-Width"))
			height, heightErr = strconv.Atoi(response.Header.Get("X-Video-Height"))
			if response.Header.Get("X-Video-Input-Mode") != "i2v" || widthErr != nil || heightErr != nil ||
				width < 128 || height < 128 || width > 1024 || height > 1024 || width%32 != 0 || height%32 != 0 || width*height > 864*480 {
				finalErr = fmt.Errorf("video service did not confirm image conditioning and valid output dimensions; text-only substitution was rejected")
				return registeredToolResult{}, finalErr
			}
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
		result, _ := json.Marshal(map[string]any{"status": "saved", "attachment": attachment, "input_mode": mode, "first_frame_image_id": args.FirstFrameImageID, "last_frame_image_id": args.LastFrameImageID, "width": width, "height": height, "frames": 124, "fps": 24, "duration_seconds": 124.0 / 24, "audio": true})
		return registeredToolResult{Result: string(result), Attachment: &attachment, AttachmentEmitted: true}, nil
	})
}

func (s *Server) videoKeyframeData(ctx context.Context, sessionID, id string) (string, error) {
	attachments, err := s.sessionImageAttachmentsForContext(ctx, sessionID)
	if err != nil {
		return "", err
	}
	item, ok := attachments[id]
	if !ok || strings.TrimSpace(id) == "" {
		return "", fmt.Errorf("keyframe image %q is not available in this conversation branch", id)
	}
	items, err := s.media.Validate([]db.Attachment{item})
	if err != nil {
		return "", err
	}
	item = items[0]
	if item.MIME != "image/png" && item.MIME != "image/jpeg" && item.MIME != "image/webp" {
		return "", fmt.Errorf("video keyframes support static PNG, JPEG or WebP images")
	}
	limit := min(s.media.Limits().ForType("image"), int64(32<<20))
	if item.Size > limit {
		return "", fmt.Errorf("video keyframe exceeds %d MiB", limit>>20)
	}
	file, err := s.media.Open(item)
	if err != nil {
		return "", err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, limit+1))
	if err != nil {
		return "", err
	}
	if int64(len(data)) > limit {
		return "", fmt.Errorf("video keyframe exceeds size limit")
	}
	info, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || info.Width <= 0 || info.Height <= 0 || int64(info.Width)*int64(info.Height) > 16<<20 {
		return "", fmt.Errorf("video keyframe must be a valid image of at most 16 megapixels")
	}
	return "data:" + item.MIME + ";base64," + base64.StdEncoding.EncodeToString(data), nil
}
