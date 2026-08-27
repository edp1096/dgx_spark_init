package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"
)

func TestControlImageRoutesToZImageBackend(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.URL.Path != "/v1/images/generations" {
			http.NotFound(w, r)
			return
		}
		var request struct {
			Model           string  `json:"model"`
			ControlImage    string  `json:"control_image"`
			ControlStrength float64 `json:"control_strength"`
			ControlStrategy string  `json:"control_strategy"`
			ControlType     string  `json:"control_type"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		decoded, err := base64.StdEncoding.DecodeString(request.ControlImage)
		if err != nil || string(decoded) != "control png" {
			t.Fatalf("control image=%q err=%v", decoded, err)
		}
		if request.Model != "z-image-test" || request.ControlStrength != 0.65 || request.ControlStrategy != "split4" || request.ControlType != "canny" {
			t.Fatalf("unexpected control request: %#v", request)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte("controlled png"))}}})
	}))
	defer worker.Close()

	cfg := config.Config{
		DataDir: t.TempDir(),
		Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}},
		Image: config.Image{
			Model: "legacy", DefaultMode: "control", DefaultWidth: 512, DefaultHeight: 512, MaxReferenceImages: 4,
			Backends: map[string]config.ImageBackend{"control": {Endpoint: worker.URL, Model: "z-image-test"}},
		},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "preserve the outline")
	_ = form.WriteField("mode", "control")
	_ = form.WriteField("control_type", "canny")
	_ = form.WriteField("control_strength", "0.65")
	part, _ := form.CreateFormFile("references", "control.png")
	_, _ = part.Write([]byte("control png"))
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			got, err := os.ReadFile(store.OutputPath(list[0].ID + ".png"))
			if err != nil || string(got) != "controlled png" {
				t.Fatalf("output=%q err=%v", got, err)
			}
			if list[0].Params["mode"] != "control" || list[0].Params["control_type"] != "canny" {
				t.Fatalf("job params=%#v", list[0].Params)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job did not complete: %#v", store.List())
}

func TestEnhancementPreservesIdentityEditContract(t *testing.T) {
	original := "Change: Replace all clothing with the outfit from the supporting reference and apply the pose from the Depth image. Preserve: identity. Do not preserve original clothing or pose."
	good := "Replace the clothing with the reference outfit and follow the Depth-controlled posture while preserving identity; do not retain the source garments or pose."
	badPose := "Use the supporting reference outfit and Depth image while preserving the original pose and clothing."
	badReference := "Change the outfit and posture while preserving identity."
	if !enhancementPreservesEditContract("edit_control", original, good) {
		t.Fatal("valid edit-control enhancement was rejected")
	}
	if enhancementPreservesEditContract("edit_control", original, badPose) {
		t.Fatal("contradictory pose preservation was accepted")
	}
	if enhancementPreservesEditContract("edit_control", original, badReference) {
		t.Fatal("missing reference and depth controls were accepted")
	}
	if !enhancementPreservesEditContract("t2i", original, badPose) {
		t.Fatal("non-edit modes must not use the edit contract")
	}
}

func TestKreaModulesRouteToCreateBackend(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/generations" {
			http.NotFound(w, r)
			return
		}
		var request struct {
			Model            string   `json:"model"`
			SourceImage      string   `json:"source_image"`
			ReferenceImages  []string `json:"reference_images"`
			ControlImage     string   `json:"control_image"`
			IdentityStrength float64  `json:"identity_strength"`
			RefBoost         float64  `json:"ref_boost"`
			SourceRefBoost   float64  `json:"source_ref_boost"`
			GroundingPX      int      `json:"grounding_px"`
			ControlStrength  float64  `json:"control_strength"`
			Style            string   `json:"style"`
			StyleStrength    float64  `json:"style_strength"`
			Styles           []struct {
				Name     string  `json:"name"`
				Strength float64 `json:"strength"`
			} `json:"styles"`
			Steps             int     `json:"steps"`
			FilterMode        string  `json:"filter_mode"`
			FilterStrength    float64 `json:"filter_strength"`
			PromptEnhancer    bool    `json:"prompt_enhancer"`
			PromptEnhStrength float64 `json:"prompt_enhancer_strength"`
			PromptTextScale   float64 `json:"prompt_text_scale"`
			IdentityFitMode   string  `json:"identity_fit_mode"`
			VAEMode           string  `json:"vae_mode"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		decode := func(value string) string {
			decoded, err := base64.StdEncoding.DecodeString(value)
			if err != nil {
				t.Fatal(err)
			}
			return string(decoded)
		}
		if request.Model != "krea-test" || decode(request.SourceImage) != "identity png" || len(request.ReferenceImages) != 2 || decode(request.ReferenceImages[0]) != "person png" || decode(request.ReferenceImages[1]) != "outfit png" || decode(request.ControlImage) != "pose png" {
			t.Fatalf("unexpected Krea images: %#v", request)
		}
		if request.IdentityStrength != 1 || request.RefBoost != 4 || request.SourceRefBoost != 1.5 || request.GroundingPX != 768 || request.ControlStrength != 0.8 || request.Style != "retroanime" || request.StyleStrength != 0.8 || request.Steps != 10 {
			t.Fatalf("unexpected Krea settings: %#v", request)
		}
		if request.FilterMode != "strong" || request.FilterStrength != 1 || !request.PromptEnhancer || request.PromptEnhStrength != 1.25 || request.PromptTextScale != 2 {
			t.Fatalf("unexpected Krea filter settings: %#v", request)
		}
		if request.IdentityFitMode != "crop" || request.VAEMode != "wan" {
			t.Fatalf("unexpected Krea identity input settings: %#v", request)
		}
		if len(request.Styles) != 2 || request.Styles[0].Name != "retroanime" || request.Styles[0].Strength != 0.8 || request.Styles[1].Name != "softwatercolor" || request.Styles[1].Strength != 0.6 {
			t.Fatalf("unexpected stacked Krea styles: %#v", request.Styles)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte("krea modules png"))}}})
	}))
	defer worker.Close()

	cfg := config.Config{
		DataDir: t.TempDir(),
		Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}},
		Image: config.Image{
			Model: "legacy", DefaultMode: "create", DefaultWidth: 1024, DefaultHeight: 1024, MaxReferenceImages: 4,
			Backends: map[string]config.ImageBackend{"create": {Endpoint: worker.URL, Model: "krea-test"}},
		},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "restage this person")
	_ = form.WriteField("mode", "create")
	_ = form.WriteField("identity_strength", "1")
	_ = form.WriteField("ref_boost", "4")
	_ = form.WriteField("source_ref_boost", "1.5")
	_ = form.WriteField("grounding_px", "768")
	_ = form.WriteField("depth_strength", "0.8")
	_ = form.WriteField("styles", `[{"name":"retroanime","strength":0.8},{"name":"softwatercolor","strength":0.6}]`)
	_ = form.WriteField("steps", "10")
	_ = form.WriteField("filter_mode", "strong")
	_ = form.WriteField("filter_strength", "1")
	_ = form.WriteField("prompt_enhancer", "true")
	_ = form.WriteField("prompt_enhancer_strength", "1.25")
	_ = form.WriteField("prompt_text_scale", "2")
	_ = form.WriteField("identity_fit_mode", "crop")
	_ = form.WriteField("vae_mode", "wan")
	for _, upload := range []struct{ field, name, content string }{
		{"identity_image", "identity.png", "identity png"},
		{"identity_reference", "person.png", "person png"},
		{"identity_reference", "outfit.png", "outfit png"},
		{"depth_image", "pose.png", "pose png"},
	} {
		part, partErr := form.CreateFormFile(upload.field, upload.name)
		if partErr != nil {
			t.Fatal(partErr)
		}
		_, _ = part.Write([]byte(upload.content))
	}
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			got, readErr := os.ReadFile(store.OutputPath(list[0].ID + ".png"))
			if readErr != nil || string(got) != "krea modules png" {
				t.Fatalf("output=%q err=%v", got, readErr)
			}
			if list[0].Params["identity"] != true || list[0].Params["depth"] != true || list[0].Params["style"] != "retroanime" {
				t.Fatalf("job params=%#v", list[0].Params)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job did not complete: %#v", store.List())
}

func TestKreaNK2ERoutesAndStoresReference(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			_ = json.NewEncoder(w).Encode(map[string]any{"status": "ok", "busy": false})
			return
		}
		var request struct {
			Model        string  `json:"model"`
			NK2EImage    string  `json:"nk2e_image"`
			NK2EMode     string  `json:"nk2e_mode"`
			NK2EStrength float64 `json:"nk2e_strength"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		decoded, err := base64.StdEncoding.DecodeString(request.NK2EImage)
		if err != nil || string(decoded) != "nk2e png" || request.Model != "krea-test" || request.NK2EMode != "canny" || request.NK2EStrength != 0.8 {
			t.Fatalf("unexpected NK2E request: %#v decoded=%q err=%v", request, decoded, err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte("nk2e result"))}}})
	}))
	defer worker.Close()

	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}}, Image: config.Image{DefaultMode: "create", DefaultWidth: 512, DefaultHeight: 512, MaxReferenceImages: 4, Backends: map[string]config.ImageBackend{"create": {Endpoint: worker.URL, Model: "krea-test"}}}}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "turn the dancer into a robot")
	_ = form.WriteField("mode", "create")
	_ = form.WriteField("nk2e_mode", "canny")
	_ = form.WriteField("nk2e_strength", "0.8")
	part, _ := form.CreateFormFile("nk2e_image", "pose.png")
	_, _ = part.Write([]byte("nk2e png"))
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			if list[0].Params["nk2e"] != true || list[0].Params["nk2e_mode"] != "canny" {
				t.Fatalf("job params=%#v", list[0].Params)
			}
			inputsReq := httptest.NewRequest(http.MethodGet, "/api/jobs/"+list[0].ID+"/inputs", nil)
			inputsReq.SetPathValue("id", list[0].ID)
			inputsRes := httptest.NewRecorder()
			handler.ServeHTTP(inputsRes, inputsReq)
			var inputs []imageJobInputInfo
			if err := json.Unmarshal(inputsRes.Body.Bytes(), &inputs); err != nil || len(inputs) != 1 || inputs[0].Role != "nk2e" {
				t.Fatalf("inputs=%#v err=%v", inputs, err)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job did not complete: %#v", store.List())
}

func TestKreaAnyPaintRoutesAndStoresSourceAndMask(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			_ = json.NewEncoder(w).Encode(map[string]any{"status": "ok", "busy": false})
			return
		}
		var request struct {
			Model        string  `json:"model"`
			Image        string  `json:"anypaint_image"`
			Mask         string  `json:"anypaint_mask"`
			Left         int     `json:"outpaint_left"`
			Right        int     `json:"outpaint_right"`
			Strength     float64 `json:"anypaint_strength"`
			Boundary     int     `json:"anypaint_boundary_redraw_px"`
			ReferenceMax int     `json:"anypaint_reference_max_edge"`
			VLMReference bool    `json:"anypaint_vlm_reference"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		image, imageErr := base64.StdEncoding.DecodeString(request.Image)
		mask, maskErr := base64.StdEncoding.DecodeString(request.Mask)
		if imageErr != nil || maskErr != nil || string(image) != "source png" || string(mask) != "mask png" {
			t.Fatalf("unexpected AnyPaint images: %#v image=%q mask=%q", request, image, mask)
		}
		if request.Model != "krea-test" || request.Left != 128 || request.Right != 384 || request.Strength != 1.1 || request.Boundary != 16 || request.ReferenceMax != 384 || !request.VLMReference {
			t.Fatalf("unexpected AnyPaint settings: %#v", request)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte("anypaint result"))}}})
	}))
	defer worker.Close()

	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}}, Image: config.Image{DefaultMode: "create", DefaultWidth: 1024, DefaultHeight: 512, MaxReferenceImages: 4, Backends: map[string]config.ImageBackend{"create": {Endpoint: worker.URL, Model: "krea-test"}}}}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "extend the room and replace the masked object")
	_ = form.WriteField("mode", "create")
	_ = form.WriteField("outpaint_left", "128")
	_ = form.WriteField("outpaint_right", "384")
	_ = form.WriteField("anypaint_strength", "1.1")
	_ = form.WriteField("anypaint_boundary_redraw_px", "16")
	for _, upload := range []struct{ field, name, content string }{
		{"anypaint_image", "source.png", "source png"},
		{"anypaint_mask", "mask.png", "mask png"},
	} {
		part, partErr := form.CreateFormFile(upload.field, upload.name)
		if partErr != nil {
			t.Fatal(partErr)
		}
		_, _ = part.Write([]byte(upload.content))
	}
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			if list[0].Params["anypaint"] != true || list[0].Params["anypaint_mask"] != true || list[0].Params["outpaint_right"] != 384 {
				t.Fatalf("job params=%#v", list[0].Params)
			}
			inputsReq := httptest.NewRequest(http.MethodGet, "/api/jobs/"+list[0].ID+"/inputs", nil)
			inputsRes := httptest.NewRecorder()
			handler.ServeHTTP(inputsRes, inputsReq)
			var inputs []imageJobInputInfo
			if err := json.Unmarshal(inputsRes.Body.Bytes(), &inputs); err != nil || len(inputs) != 2 || inputs[0].Role != "anypaint" || inputs[1].Role != "anypaint_mask" {
				t.Fatalf("inputs=%#v err=%v", inputs, err)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("AnyPaint job did not complete: %#v", store.List())
}

func TestKreaAnyPaintOutpaintAllowsBlankPrompt(t *testing.T) {
	const automaticPrompt = "Extend the original image naturally into a complete, coherent composition while preserving its subjects, style, lighting, perspective, and visual continuity."
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			_ = json.NewEncoder(w).Encode(map[string]any{"status": "ok", "busy": false})
			return
		}
		var request struct {
			Prompt string `json:"prompt"`
			Image  string `json:"anypaint_image"`
			Right  int    `json:"outpaint_right"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		image, err := base64.StdEncoding.DecodeString(request.Image)
		if err != nil || string(image) != "source png" || request.Prompt != automaticPrompt || request.Right != 256 {
			t.Fatalf("unexpected blank-prompt Outpaint request: %#v image=%q err=%v", request, image, err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte("outpaint result"))}}})
	}))
	defer worker.Close()

	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}}, Image: config.Image{DefaultMode: "create", DefaultWidth: 1024, DefaultHeight: 512, MaxReferenceImages: 4, Backends: map[string]config.ImageBackend{"create": {Endpoint: worker.URL, Model: "krea-test"}}}}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("mode", "create")
	_ = form.WriteField("outpaint_right", "256")
	part, err := form.CreateFormFile("anypaint_image", "source.png")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = part.Write([]byte("source png"))
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			if list[0].Prompt != automaticPrompt || list[0].Params["anypaint_mask"] != false {
				t.Fatalf("job=%#v", list[0])
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("blank-prompt Outpaint job did not complete: %#v", store.List())
}

func TestKreaStyleReferenceRoutesAndIsReusable(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			_ = json.NewEncoder(w).Encode(map[string]any{"status": "ok", "busy": false})
			return
		}
		var request struct {
			Images   []string `json:"style_reference_images"`
			Strength float64  `json:"style_reference_strength"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		decoded, err := base64.StdEncoding.DecodeString(request.Images[0])
		if err != nil || string(decoded) != "style reference" || request.Strength != 0.9 {
			t.Fatalf("unexpected style reference request: %#v decoded=%q err=%v", request, decoded, err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte("styled png"))}}})
	}))
	defer worker.Close()
	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}}, Image: config.Image{DefaultMode: "create", DefaultWidth: 1024, DefaultHeight: 1024, MaxReferenceImages: 4, Backends: map[string]config.ImageBackend{"create": {Endpoint: worker.URL, Model: "krea-test"}}}}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "use this visual style")
	_ = form.WriteField("mode", "create")
	_ = form.WriteField("style_reference_strength", "0.9")
	part, _ := form.CreateFormFile("style_reference_images", "style.png")
	_, _ = part.Write([]byte("style reference"))
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			inputsReq := httptest.NewRequest(http.MethodGet, "/api/jobs/"+list[0].ID+"/inputs", nil)
			inputsRes := httptest.NewRecorder()
			handler.ServeHTTP(inputsRes, inputsReq)
			var inputs []imageJobInputInfo
			if json.Unmarshal(inputsRes.Body.Bytes(), &inputs) != nil || len(inputs) != 1 || inputs[0].Role != "style_reference" {
				t.Fatalf("unexpected stored inputs: %s", inputsRes.Body.String())
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("style reference job did not complete")
}
