package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	mediaprompt "mediaapp/internal/prompt"
)

const describeSystem = `You are a precise visual analyst. Describe only what is visibly present in the supplied image. Include the subject count, identities or object types, colors, materials, spatial relationships, composition, viewpoint, pose, medium or photographic style, and all readable text exactly. Do not infer hidden content or add objects. Output one English paragraph of 90-160 words with no heading or preamble.`

type visionCase struct {
	ID       string   `json:"id"`
	Image    string   `json:"image"`
	Motion   string   `json:"motion"`
	Required []string `json:"required"`
	Exact    []string `json:"exact,omitempty"`
}

type result struct {
	Case      visionCase `json:"case"`
	Task      string     `json:"task"`
	Output    string     `json:"output"`
	LatencyMS int64      `json:"latency_ms"`
	Words     int        `json:"words"`
	Hits      int        `json:"hits"`
	Total     int        `json:"total"`
	Missing   []string   `json:"missing,omitempty"`
	Error     string     `json:"error,omitempty"`
}

type report struct {
	Engine      string    `json:"engine"`
	Endpoint    string    `json:"endpoint"`
	Model       string    `json:"model"`
	GeneratedAt time.Time `json:"generated_at"`
	Results     []result  `json:"results"`
	Hits        int       `json:"hits"`
	Total       int       `json:"total"`
	MeanMS      float64   `json:"mean_latency_ms"`
}

func main() {
	engine := flag.String("engine", "vision-engine", "engine name")
	endpoint := flag.String("endpoint", "http://127.0.0.1:8696", "OpenAI-compatible API base URL")
	model := flag.String("model", "huihui-gemma4-e2b", "model or alias")
	root := flag.String("asset-root", "web/dist", "Media web asset root")
	convertedJPEG := flag.Bool("converted-jpeg", false, "use .jpg versions of all benchmark assets")
	output := flag.String("output", "vision-benchmark.json", "JSON report path")
	flag.Parse()

	client := &http.Client{Timeout: 5 * time.Minute}
	rep := report{Engine: *engine, Endpoint: *endpoint, Model: *model, GeneratedAt: time.Now()}
	var totalLatency int64
	for _, tc := range suite(*root, *convertedJPEG) {
		for _, task := range []string{"describe", "i2v"} {
			r, err := run(client, *endpoint, *model, tc, task)
			if err != nil {
				r.Error = err.Error()
			}
			score(&r)
			rep.Hits += r.Hits
			rep.Total += r.Total
			totalLatency += r.LatencyMS
			rep.Results = append(rep.Results, r)
			fmt.Printf("%-22s %-8s %2d/%-2d %4d words %6d ms", tc.ID, task, r.Hits, r.Total, r.Words, r.LatencyMS)
			if r.Error != "" {
				fmt.Printf(" ERROR: %s", r.Error)
			}
			fmt.Println()
		}
	}
	if len(rep.Results) > 0 {
		rep.MeanMS = float64(totalLatency) / float64(len(rep.Results))
	}
	data, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(*output, append(data, '\n'), 0o644); err != nil {
		panic(err)
	}
	fmt.Printf("TOTAL %d/%d (%.1f%%), mean %.0f ms -> %s\n", rep.Hits, rep.Total, 100*float64(rep.Hits)/float64(rep.Total), rep.MeanMS, *output)
}

func suite(root string, convertedJPEG bool) []visionCase {
	p := func(parts ...string) string { return filepath.Join(append([]string{root}, parts...)...) }
	asset := func(directory, name, extension string) string {
		if convertedJPEG {
			extension = ".jpg"
		}
		return p(directory, name+extension)
	}
	return []visionCase{
		{ID: "cat-bento", Image: asset("prompt-examples", "sogni-cat-bento", ".webp"), Motion: "The camera remains static. One of the cat's rice paws makes a tiny wave, while the food arrangement stays unchanged.", Required: []string{"cat", "rice", "nori", "egg", "tomato", "broccoli", "bento"}},
		{ID: "animal-tower", Image: asset("prompt-examples", "sogni-animal-tower", ".webp"), Motion: "The camera remains static. The two flying birds flap their wings while the stacked animals hold their positions.", Required: []string{"giraffe", "hippopotamus", "zebra", "lion", "lemur", "parrot", "circus", "crowd"}},
		{ID: "ocean-illustration", Image: asset("prompt-examples", "official-ocean", ".webp"), Motion: "The camera remains static. The person takes one careful step through the shallow water as the small waves move naturally.", Required: []string{"person", "water", "orange", "green", "rock", "peach", "illustration"}},
		{ID: "tree-person-dog", Image: asset("prompt-examples", "official-tree-dog", ".webp"), Motion: "The camera remains static. The small white dog turns its head slightly while the seated person remains still.", Required: []string{"tree", "person", "white dog", "shade", "grass", "hill", "blue", "yellow"}},
		{ID: "cat-ocr", Image: asset("prompt-examples", "sogni-cat-typography", ".webp"), Motion: "The camera remains static. The black-and-white cat blinks once without changing its position.", Required: []string{"black", "white", "cat", "face"}, Exact: []string{"CAT"}},
		{ID: "gallery-ocr", Image: asset("prompt-examples", "sogni-gallery-long-exposure", ".webp"), Motion: "The camera remains static. The woman in front of the painting remains still while the blurred visitors continue moving past her.", Required: []string{"gallery", "woman", "red dress", "painting", "blur", "visitors"}, Exact: []string{"13B", "KREA 2 Turbo"}},
		{ID: "smoky-makeup", Image: asset("makeup-library", "02-smoky", ".jpg"), Motion: "The camera remains static. The woman slowly blinks once while keeping the same neutral expression.", Required: []string{"woman", "dark hair", "smoky", "eye", "neutral", "gray"}},
		{ID: "pose-ocr", Image: asset(filepath.Join("pose-library", "images"), "pose_087", ".webp"), Motion: "The camera remains static. The woman begins to rise very slightly from her initial pose.", Required: []string{"woman", "side", "squat", "hands", "yellow", "white"}},
	}
}

func run(client *http.Client, endpoint, model string, tc visionCase, task string) (result, error) {
	data, err := os.ReadFile(tc.Image)
	if err != nil {
		return result{Case: tc, Task: task}, err
	}
	mimeType := mime.TypeByExtension(strings.ToLower(filepath.Ext(tc.Image)))
	if mimeType == "" {
		mimeType = http.DetectContentType(data)
	}
	userText := "Describe the supplied image precisely."
	system := describeSystem
	if task == "i2v" {
		userText = "User Raw Input Prompt: " + tc.Motion
		system = mediaprompt.System("i2v", true)
	}
	payload := map[string]any{
		"model": model,
		"messages": []map[string]any{
			{"role": "system", "content": system},
			{"role": "user", "content": []map[string]any{
				{"type": "image_url", "image_url": map[string]string{"url": "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data)}},
				{"type": "text", "text": userText},
			}},
		},
		"max_completion_tokens": 600,
		"temperature":           0,
		"top_k":                 1,
		"seed":                  42,
		"reasoning_effort":      "none",
	}
	body, _ := json.Marshal(payload)
	started := time.Now()
	resp, err := client.Post(strings.TrimRight(endpoint, "/")+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	r := result{Case: tc, Task: task, LatencyMS: time.Since(started).Milliseconds()}
	if err != nil {
		return r, err
	}
	defer resp.Body.Close()
	responseData, err := io.ReadAll(resp.Body)
	r.LatencyMS = time.Since(started).Milliseconds()
	if err != nil {
		return r, err
	}
	if resp.StatusCode/100 != 2 {
		return r, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(responseData)))
	}
	var decoded struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(responseData, &decoded); err != nil {
		return r, err
	}
	if len(decoded.Choices) == 0 {
		return r, fmt.Errorf("response has no choices")
	}
	r.Output = strings.TrimSpace(decoded.Choices[0].Message.Content)
	r.Words = len(strings.Fields(r.Output))
	return r, nil
}

func score(r *result) {
	lower := strings.ToLower(r.Output)
	for _, required := range r.Case.Required {
		r.Total++
		if strings.Contains(lower, strings.ToLower(required)) {
			r.Hits++
		} else {
			r.Missing = append(r.Missing, required)
		}
	}
	for _, exact := range r.Case.Exact {
		r.Total++
		if strings.Contains(r.Output, exact) {
			r.Hits++
		} else {
			r.Missing = append(r.Missing, exact)
		}
	}
}
