package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
)

const Profile = "google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b:text:bf16:768:chunk512-o48-v1"
const Dimensions = 768

type Input struct {
	Title string `json:"title,omitempty"`
	Text  string `json:"text"`
}
type Segment struct {
	InputIndex int       `json:"input_index"`
	Segment    int       `json:"segment"`
	Text       string    `json:"text"`
	Vector     []float32 `json:"embedding"`
}
type Response struct {
	Profile    string    `json:"profile"`
	Dimensions int       `json:"dimensions"`
	Data       []Segment `json:"data"`
}

func Encode(ctx context.Context, endpoint, task string, inputs []Input) (Response, error) {
	var out Response
	b, err := json.Marshal(map[string]any{"task": task, "inputs": inputs})
	if err != nil {
		return out, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(endpoint, "/")+"/v1/encode", bytes.NewReader(b))
	if err != nil {
		return out, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return out, fmt.Errorf("embedding HTTP %d: %s", resp.StatusCode, detail)
	}
	if err = json.NewDecoder(io.LimitReader(resp.Body, 16<<20)).Decode(&out); err != nil {
		return out, err
	}
	if out.Profile != Profile || out.Dimensions != Dimensions || len(out.Data) == 0 {
		return out, fmt.Errorf("incompatible embedding profile")
	}
	seen := make(map[int]bool)
	for _, p := range out.Data {
		if p.InputIndex < 0 || p.InputIndex >= len(inputs) || len(p.Vector) != Dimensions || p.Segment < 0 || p.Text == "" {
			return out, fmt.Errorf("invalid embedding response")
		}
		norm := 0.0
		for _, v := range p.Vector {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return out, fmt.Errorf("non-finite embedding")
			}
			norm += float64(v) * float64(v)
		}
		if norm < 0.99 || norm > 1.01 {
			return out, fmt.Errorf("unnormalized embedding")
		}
		seen[p.InputIndex] = true
	}
	if len(seen) != len(inputs) {
		return out, fmt.Errorf("missing embedding input")
	}
	if task == "query" && len(out.Data) != 1 {
		return out, fmt.Errorf("invalid query embedding")
	}
	return out, nil
}
