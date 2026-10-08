package imagegen

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"time"
)

type progressRequestKey struct{}

// WithProgressRequest correlates supported services without changing their JSON API.
func WithProgressRequest(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, progressRequestKey{}, id)
}

type Client struct {
	endpoint string
	model    string
	http     *http.Client
}

type LoRA struct {
	Filename string `json:"filename"`
	Size     int64  `json:"size,omitempty"`
}

type Capabilities struct {
	Model      string   `json:"model"`
	Styles     []string `json:"styles"`
	Operations []string `json:"operations"`
	UserLoRAs  []LoRA   `json:"user_loras"`
}

type Result struct {
	Image   []byte
	Seed    int64
	Control []byte
}

type ReferenceImage struct {
	Data []byte
	MIME string
}

// Edit sends ordered references to the configured image endpoint.
// A failed edit is never retried as text-only generation.
func (c *Client) Edit(ctx context.Context, images []ReferenceImage, prompt, size string, seed *int64, operation ...string) (Result, error) {
	if len(images) < 1 || len(images) > 4 {
		return Result{}, errors.New("image editing requires one to four reference images")
	}
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	for i, item := range images {
		ext := map[string]string{"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[item.MIME]
		if ext == "" || len(item.Data) == 0 {
			return Result{}, errors.New("reference images must be nonempty PNG, JPEG, or WebP files")
		}
		part, err := form.CreateFormFile("image", fmt.Sprintf("reference-%d%s", i+1, ext))
		if err != nil {
			return Result{}, err
		}
		if _, err = part.Write(item.Data); err != nil {
			return Result{}, err
		}
	}
	fields := map[string]string{"model": c.model, "prompt": prompt, "size": size, "n": "1", "response_format": "b64_json"}
	if len(operation) > 0 {
		fields["operation"] = operation[0]
	}
	if seed != nil {
		fields["seed"] = fmt.Sprint(*seed)
	}
	for key, value := range fields {
		if err := form.WriteField(key, value); err != nil {
			return Result{}, err
		}
	}
	if err := form.Close(); err != nil {
		return Result{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint+"/v1/images/edits", &body)
	if err != nil {
		return Result{}, err
	}
	req.Header.Set("Content-Type", form.FormDataContentType())
	resp, err := c.http.Do(req)
	if err != nil {
		return Result{}, fmt.Errorf("reference image API request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return Result{}, fmt.Errorf("reference image API returned HTTP %d: %s", resp.StatusCode, responseError(resp))
	}
	var output struct {
		Seed int64 `json:"seed"`
		Data []struct {
			Base64 string `json:"b64_json"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 32<<20)).Decode(&output); err != nil {
		return Result{}, err
	}
	if len(output.Data) == 0 || output.Data[0].Base64 == "" {
		return Result{}, errors.New("reference image API response did not contain an image")
	}
	data, err := base64.StdEncoding.DecodeString(output.Data[0].Base64)
	if err != nil {
		return Result{}, err
	}
	return Result{Image: data, Seed: output.Seed}, nil
}

type SegmentResult struct {
	Mask   []byte
	Boxes  [][]float64
	Scores []float64
}

func New(endpoint, model string, timeout time.Duration) *Client {
	if timeout <= 0 {
		timeout = 30 * time.Minute
	}
	return &Client{endpoint: strings.TrimRight(endpoint, "/"), model: model, http: &http.Client{Timeout: timeout}}
}

func (c *Client) Health(ctx context.Context) map[string]any {
	status := map[string]any{"status": "offline", "endpoint": c.endpoint, "model": c.model}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.endpoint+"/health", nil)
	if err != nil {
		status["error"] = err.Error()
		return status
	}
	resp, err := c.http.Do(req)
	if err != nil {
		status["error"] = err.Error()
		return status
	}
	defer resp.Body.Close()
	status["status"] = "ok"
	if resp.StatusCode >= 300 {
		status["status"] = "degraded"
		status["error"] = responseError(resp)
	}
	return status
}

func (c *Client) Capabilities(ctx context.Context) (Capabilities, error) {
	result := Capabilities{
		Model:      c.model,
		Styles:     []string{"darkbrush", "dotmatrix", "kidsdrawing", "neondrip", "rainywindow", "retroanime", "softwatercolor", "sunsetblur", "vintagetarot"},
		Operations: []string{"generate", "identity_edit", "depth", "vision_reference", "style_reference", "nk2e_edit", "nk2e_canny", "inpaint", "outpaint", "detail_enhance", "video_keyframes", "sprite_8way"},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.endpoint+"/v1/loras", nil)
	if err != nil {
		return result, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		// Basic image APIs do not expose the optional LoRA catalogue. The
		// built-in operation/style catalogue remains usable in that case.
		return result, nil
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return result, nil
	}
	if resp.StatusCode >= 300 {
		return result, fmt.Errorf("image LoRA catalogue: %s", responseError(resp))
	}
	var body struct {
		Data       []LoRA    `json:"data"`
		Model      string    `json:"model"`
		Operations *[]string `json:"operations"`
		Styles     *[]string `json:"styles"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&body); err != nil {
		return result, fmt.Errorf("decode image LoRA catalogue: %w", err)
	}
	result.UserLoRAs = body.Data
	if body.Model != "" {
		result.Model = body.Model
	}
	if body.Operations != nil {
		result.Operations = *body.Operations
	}
	if body.Styles != nil {
		result.Styles = *body.Styles
	}
	return result, nil
}

func (c *Client) Generate(ctx context.Context, payload map[string]any) (Result, error) {
	payload["model"] = c.model
	payload["n"] = 1
	payload["response_format"] = "b64_json"
	payload["output_format"] = "png"
	encoded, err := json.Marshal(payload)
	if err != nil {
		return Result{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint+"/v1/images/generations", bytes.NewReader(encoded))
	if err != nil {
		return Result{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if id, _ := ctx.Value(progressRequestKey{}).(string); id != "" {
		req.Header.Set("X-SparkTalk-Request-ID", id)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return Result{}, fmt.Errorf("image API request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return Result{}, fmt.Errorf("image API returned HTTP %d: %s", resp.StatusCode, responseError(resp))
	}
	var body struct {
		Seed int64 `json:"seed"`
		Data []struct {
			Base64 string `json:"b64_json"`
		} `json:"data"`
		Control string `json:"control_b64_json"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 32<<20)).Decode(&body); err != nil {
		return Result{}, fmt.Errorf("decode image API response: %w", err)
	}
	if len(body.Data) == 0 || body.Data[0].Base64 == "" {
		return Result{}, fmt.Errorf("image API response did not contain an image")
	}
	image, err := base64.StdEncoding.DecodeString(body.Data[0].Base64)
	if err != nil {
		return Result{}, fmt.Errorf("decode image API image: %w", err)
	}
	var control []byte
	if body.Control != "" {
		control, _ = base64.StdEncoding.DecodeString(body.Control)
	}
	return Result{Image: image, Seed: body.Seed, Control: control}, nil
}

func (c *Client) Segment(ctx context.Context, imageDataURL, prompt string) (SegmentResult, error) {
	payload, _ := json.Marshal(map[string]any{"image": imageDataURL, "prompt": prompt, "grow": 8, "feather": 4})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint+"/v1/masks/segment", bytes.NewReader(payload))
	if err != nil {
		return SegmentResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return SegmentResult{}, fmt.Errorf("automatic mask request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return SegmentResult{}, fmt.Errorf("automatic mask returned HTTP %d: %s", resp.StatusCode, responseError(resp))
	}
	var body struct {
		Mask   string      `json:"mask_b64_json"`
		Boxes  [][]float64 `json:"boxes"`
		Scores []float64   `json:"scores"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 16<<20)).Decode(&body); err != nil {
		return SegmentResult{}, err
	}
	mask, err := base64.StdEncoding.DecodeString(body.Mask)
	if err != nil || len(mask) == 0 {
		return SegmentResult{}, errors.New("automatic mask response did not contain a valid mask")
	}
	return SegmentResult{Mask: mask, Boxes: body.Boxes, Scores: body.Scores}, nil
}

func responseError(resp *http.Response) string {
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	text := strings.TrimSpace(string(data))
	if text == "" {
		return resp.Status
	}
	return text
}
