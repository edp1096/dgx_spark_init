package server

import (
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"mediaapp/internal/config"
)

func TestNormalizeImageOptionsAcceptsOfficialINT8AndDisablesFilterForUserLoRA(t *testing.T) {
	form := url.Values{
		"user_loras": {`[{"filename":"face.safetensors","strength":0.8}]`},
	}
	request := httptest.NewRequest("POST", "/api/images", strings.NewReader(form.Encode()))
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if err := request.ParseForm(); err != nil {
		t.Fatal(err)
	}
	options := imageGenerationOptions{
		checkpoint: "official-int8", filterMode: "balanced", filterStrength: 1,
		promptTextScale: 1.75,
	}
	if err := normalizeImageGenerationOptions(request, config.Config{}, &options); err != nil {
		t.Fatal(err)
	}
	if options.filterMode != "off" || options.filterStrength != 0 {
		t.Fatalf("user LoRA must disable filter bypass, got mode=%q strength=%v", options.filterMode, options.filterStrength)
	}
}
