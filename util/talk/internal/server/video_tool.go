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

const videoToolSystemPrompt = `Use video_generate only when the user requests a video. Never replace a requested still image with a video or suggest extracting a frame as if that fulfilled the image request. For a still image use image_generate; if it is unavailable, report that limitation without generating another modality. MiniMax H3 generates a 5.17-second MP4 with audio. Write a detailed English prompt preserving the requested subject, motion and sounds. Image-to-video is supported: when the user asks to animate an image or use it as frame 0, supply its actual conversation attachment ID as first_frame_image_id. Optional last_frame_image_id anchors the final frame. These images are encoded into the video model, not merely described in a text prompt. Do not substitute text-only generation for a requested image-conditioned video. Use only listed image attachment IDs, never a video ID. Inspect the selected image to write a suitable motion prompt. Image-conditioned output automatically follows the source aspect ratio; use the returned dimensions. Set resolution="720p" when the user requests 720p; text-only output is then natively generated at 1280x720. The default resolution="480p" produces 864x480. For image-conditioned generation, the selected resolution sets the pixel budget and the source aspect ratio is preserved; report the actual returned dimensions. All output is 24fps. Do not refuse 720p or substitute upscaling. Arbitrary duration and reference video/audio inputs are unavailable. 480p generation usually takes about a minute after startup; 720p takes longer. Success is already stored and attached; never redownload or repeat a successful call. Describe the result briefly without claiming visual inspection of the generated video.`

func (s *Server) registerVideoTool(ctx context.Context, reg *completionToolRegistry, cfg config.Config, sink mediaAttachmentSink) {
	endpoint, ok := videoRuntime(cfg)
	if !ok || sink == nil {
		return
	}
	reg.prompts = append(reg.prompts, videoToolSystemPrompt, imageAttachmentCatalogForContext(ctx, s, reg.sessionID))
	reg.register(llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "video_generate", Description: "Generate and attach a 5.17-second MiniMax H3 video with audio at 24fps. Supports text-to-video and image-to-video using actual conversation image IDs as first/last frame anchors. Supports native 720p via resolution. Image-conditioned size follows the source aspect ratio. 480p generation usually takes about a minute after startup; 720p takes longer.", Parameters: json.RawMessage(`{"type":"object","properties":{"prompt":{"type":"string","minLength":1,"maxLength":8192},"resolution":{"type":"string","enum":["480p","720p"],"description":"Use 720p when requested: native 1280x720 for text-to-video. Default 480p is 864x480. Image-to-video preserves source aspect within the selected pixel budget."},"seed":{"type":"integer","minimum":0,"maximum":9223372036854775807},"first_frame_image_id":{"type":"string","minLength":1,"description":"Actual conversation image attachment ID to encode as the first frame (frame 0). Required when animating an input image."},"last_frame_image_id":{"type":"string","minLength":1,"description":"Optional conversation image attachment ID to encode as the final frame (frame 123)."}},"required":["prompt"],"additionalProperties":false}`)}}, func(ctx context.Context, call llm.ToolCall, messages []llm.Message, emit eventEmitter) (registeredToolResult, error) {
		userRequest := currentTitleRequestText(messages)
		if turn := turnFrom(ctx); turn != nil && turn.session == reg.sessionID {
			_, userRequest = s.clipTurnIntent(ctx, reg.sessionID)
		}
		if standaloneImageRequest(userRequest) {
			return registeredToolResult{}, fmt.Errorf("정지 이미지 요청을 영상으로 대체할 수 없습니다. image_generate를 사용하세요. 이미지 도구가 없으면 현재 설정에서 사용할 수 없다고 안내하세요")
		}

		var args struct {
			Prompt            string  `json:"prompt"`
			Resolution        string  `json:"resolution,omitempty"`
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
		if args.Resolution != "" && args.Resolution != "480p" && args.Resolution != "720p" {
			return registeredToolResult{}, fmt.Errorf("video resolution must be 480p or 720p")
		}
		payload := map[string]any{"prompt": args.Prompt, "model": "minimax-h3-nvfp4"}
		if args.Resolution != "" {
			payload["resolution"] = args.Resolution
		}
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
		}
		if mode == "i2v" || args.Resolution == "720p" || response.Header.Get("X-Video-Width") != "" {
			var widthErr, heightErr error
			width, widthErr = strconv.Atoi(response.Header.Get("X-Video-Width"))
			height, heightErr = strconv.Atoi(response.Header.Get("X-Video-Height"))
			grid, maxEdge, maxArea := 32, 1024, 864*480
			expectedWidth, expectedHeight := 864, 480
			if args.Resolution == "720p" {
				grid, maxEdge, maxArea = 16, 1280, 1280*720
				expectedWidth, expectedHeight = 1280, 720
			}
			if mode == "i2v" {
				grid = 32
			}
			if response.Header.Get("X-Video-Input-Mode") != mode || widthErr != nil || heightErr != nil ||
				width < 128 || height < 128 || width > maxEdge || height > maxEdge || width%grid != 0 || height%grid != 0 || width*height > maxArea ||
				(mode == "t2v" && (width != expectedWidth || height != expectedHeight)) {
				finalErr = fmt.Errorf("video service did not confirm the requested input mode and resolution; substitution was rejected")
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

// A narrow guard for explicit still-image requests. Ambiguous follow-ups retain
// their normal routing; mentioning an input image does not block image-to-video.
func standaloneImageRequest(request string) bool {
	text := strings.ToLower(userURLPattern.ReplaceAllString(request, " "))
	still := false
	for _, word := range []string{"이미지", "그림", "사진", "그려", "초상화", "일러스트", "썸네일", "포스터", "draw", "sketch", "painting", "image", "picture", "photo", "portrait", "illustration", ".png", ".jpg"} {
		still = still || strings.Contains(text, word)
	}
	if !still {
		return false
	}
	for _, negative := range []string{"영상 말고", "영상말고", "동영상 말고", "영상은 말고", "영상 만들지 말고", "영상 만들지 마", "영상은 만들지 마", "no video", "not a video", "without video"} {
		text = strings.ReplaceAll(text, negative, " ")
	}
	for _, style := range []string{"애니메이션풍", "애니메이션 스타일", "animation style", "animated style", "video game"} {
		text = strings.ReplaceAll(text, style, " ")
	}
	for _, word := range []string{"영상", "동영상", "클립", "움직", "애니메이션", "쇼츠", "video", "animate", "animation", "moving", "motion", "shorts"} {
		if strings.Contains(text, word) {
			return false
		}
	}
	return true
}
