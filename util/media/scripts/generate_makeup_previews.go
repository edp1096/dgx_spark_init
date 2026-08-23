package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	_ "image/png"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

type preset struct {
	ID     string `json:"id"`
	Prompt string `json:"prompt"`
	Image  string `json:"image"`
}

type generationResponse struct {
	Data []struct {
		B64JSON string `json:"b64_json"`
	} `json:"data"`
}

const basePrompt = "A centered close-up beauty portrait of the same androgynous adult model, symmetrical face, short dark hair tucked away from the face, bare shoulders, looking directly into the camera, neutral grey studio background, soft even beauty-dish lighting, realistic skin texture, 85mm lens, identical framing. Face styling: "

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: generate_makeup_previews PRESETS_JSON OUTPUT_DIR")
		os.Exit(2)
	}
	raw, err := os.ReadFile(os.Args[1])
	check(err)
	var presets []preset
	check(json.Unmarshal(raw, &presets))
	check(os.MkdirAll(os.Args[2], 0o755))

	client := &http.Client{Timeout: 10 * time.Minute}
	for index, item := range presets {
		output := filepath.Join(os.Args[2], item.Image)
		if _, err := os.Stat(output); err == nil {
			fmt.Printf("[%02d/%02d] exists %s\n", index+1, len(presets), item.Image)
			continue
		}
		payload, err := json.Marshal(map[string]any{
			"prompt": basePrompt + item.Prompt + ".",
			"model":  "krea2-turbo-nvfp4", "n": 1, "size": "512x512", "seed": 260821,
			"response_format": "b64_json", "steps": 8, "filter_mode": "off",
		})
		check(err)
		fmt.Printf("[%02d/%02d] generating %s\n", index+1, len(presets), item.Image)
		var response generationResponse
		for attempt := 1; attempt <= 3; attempt++ {
			response, err = generate(client, payload)
			if err == nil && len(response.Data) > 0 && response.Data[0].B64JSON != "" {
				break
			}
			fmt.Printf("  attempt %d failed: %v\n", attempt, err)
			time.Sleep(time.Duration(attempt) * 2 * time.Second)
		}
		if err != nil || len(response.Data) == 0 || response.Data[0].B64JSON == "" {
			check(fmt.Errorf("generation failed for %s: %w", item.ID, err))
		}
		encoded, err := base64.StdEncoding.DecodeString(response.Data[0].B64JSON)
		check(err)
		decoded, _, err := image.Decode(bytes.NewReader(encoded))
		check(err)
		file, err := os.Create(output)
		check(err)
		check(jpeg.Encode(file, decoded, &jpeg.Options{Quality: 84}))
		check(file.Close())
	}
}

func generate(client *http.Client, payload []byte) (generationResponse, error) {
	request, err := http.NewRequest(http.MethodPost, "http://127.0.0.1:8691/v1/images/generations", bytes.NewReader(payload))
	if err != nil {
		return generationResponse{}, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return generationResponse{}, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return generationResponse{}, err
	}
	if response.StatusCode != http.StatusOK {
		return generationResponse{}, fmt.Errorf("HTTP %d: %s", response.StatusCode, body)
	}
	var result generationResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return result, err
	}
	return result, nil
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
