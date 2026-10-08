package server

import (
	"bytes"
	"encoding/json"
	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestStoryboardPlannerReturnsStandaloneScenes(t *testing.T) {
	promptEngine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if r.URL.Path != "/v1/chat/completions" || json.NewDecoder(r.Body).Decode(&request) != nil || len(request.Messages) != 2 {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if !strings.Contains(request.Messages[0].Content, "canonical English visual identity block") || !strings.Contains(request.Messages[1].Content, "시장") {
			t.Fatalf("unexpected planning messages: %#v", request.Messages)
		}
		content := `{"shared_prompt":"검은 단발과 남색 코트를 입은 한국인 탐정, 비 내리는 누아르 도시","world_prompt_en":"A cinematic Korean noir world with realistic materials and subdued colors.","characters":[{"id":"detective","name_ko":"탐정","name_en":"Mina","description_ko":"검은 단발과 남색 코트","prompt_en":"A Korean detective named Mina with a short black bob, oval face, dark brown eyes, and a navy wool coat."}],"scenes":[{"scene_title":"단서 발견","original_prompt":"wrong","character_ids":["detective"],"scene_prompt_en":"Mina finds a clue in a rainy market, medium-wide cinematic framing.","strategy":"major","change_summary":"시장 단서"},{"scene_title":"옥상 대면","original_prompt":"wrong","character_ids":["detective"],"scene_prompt_en":"Mina confronts a suspect on a rooftop at night, low-angle composition.","strategy":"minor","change_summary":"옥상 대면"}]}`
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": content}}}})
	}))
	defer promptEngine.Close()

	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"prompt": {Endpoint: promptEngine.URL}}, PromptEnhancement: config.PromptEnhancement{Model: "planner", MaxTokens: 600}}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()
	body := bytes.NewBufferString(`{"prompts":["탐정이 시장에서 단서를 찾는다","탐정이 옥상에서 범인과 대면한다"]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/prompts/sequence-plan", body)
	req.Header.Set("Content-Type", "application/json")
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	var plan imageSequencePromptPlan
	if err := json.Unmarshal(res.Body.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	if len(plan.Scenes) != 2 || plan.Scenes[0].Strategy != "storyboard" || plan.Scenes[1].Strategy != "storyboard" {
		t.Fatalf("unexpected strategies: %#v", plan.Scenes)
	}
	if plan.Scenes[0].OriginalPrompt != "탐정이 시장에서 단서를 찾는다" || plan.Scenes[1].SceneTitle != "옥상 대면" {
		t.Fatalf("unexpected scenes: %#v", plan.Scenes)
	}
	identity := plan.Characters[0].PromptEN
	if !strings.Contains(plan.Scenes[0].EnhancedPrompt, identity) || !strings.Contains(plan.Scenes[1].EnhancedPrompt, identity) {
		t.Fatalf("canonical identity was not injected verbatim: %#v", plan.Scenes)
	}
	if strings.Contains(plan.Scenes[1].EnhancedPrompt, "same detective") {
		t.Fatalf("scene retained a relative identity reference: %q", plan.Scenes[1].EnhancedPrompt)
	}
}

func TestStoryboardPlannerCanSplitStoryOutline(t *testing.T) {
	promptEngine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		content := `{"shared_prompt":"주황색 배달 로봇과 따뜻한 미래 도시","world_prompt_en":"A warm cinematic retro-futuristic city.","characters":[{"id":"robot","name_ko":"배달 로봇","name_en":"Rho","description_ko":"둥근 주황색 배달 로봇","prompt_en":"A compact orange delivery robot named Rho with a round cream faceplate, two blue circular eyes, a short antenna, and sturdy silver joints."}],"scenes":[{"scene_title":"출발","original_prompt":"로봇이 수리점에서 출발한다","character_ids":["robot"],"scene_prompt_en":"Rho leaves a repair shop at dawn.","strategy":"storyboard","change_summary":"출발"},{"scene_title":"시장","original_prompt":"로봇이 시장을 탐색한다","character_ids":["robot"],"scene_prompt_en":"Rho searches a crowded neon market.","strategy":"storyboard","change_summary":"탐색"},{"scene_title":"재회","original_prompt":"로봇이 주인과 재회한다","character_ids":["robot"],"scene_prompt_en":"Rho reunites with its owner on a rooftop garden.","strategy":"storyboard","change_summary":"재회"}]}`
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": content}}}})
	}))
	defer promptEngine.Close()
	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"prompt": {Endpoint: promptEngine.URL}}, PromptEnhancement: config.PromptEnhancement{Model: "planner"}}
	store, _ := jobs.New(cfg.DataDir)
	res := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/prompts/sequence-plan", bytes.NewBufferString(`{"outline":"로봇이 주인을 찾는다","scene_count":3}`))
	req.Header.Set("Content-Type", "application/json")
	New(cfg, store, nil).Handler().ServeHTTP(res, req)
	if res.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	var plan imageSequencePromptPlan
	_ = json.Unmarshal(res.Body.Bytes(), &plan)
	if len(plan.Scenes) != 3 || plan.Scenes[2].OriginalPrompt == "" {
		t.Fatalf("outline was not split: %#v", plan)
	}
	if plan.CanonicalPrompt == "" || plan.Scenes[0].EnhancedPrompt == plan.Scenes[0].ScenePromptEN {
		t.Fatalf("canonical prompt was not compiled: %#v", plan)
	}
}

func TestStoryboardPlannerTreatsEditedKoreanBibleAsAuthoritative(t *testing.T) {
	promptEngine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		content := `{"shared_prompt":"모델이 바꾼 설정","world_prompt_en":"Soft watercolor storybook style.","characters":[{"id":"robot","name_ko":"로봇","name_en":"Sol","description_ko":"노란 로봇","prompt_en":"A small yellow robot named Sol with a square faceplate and one green eye."}],"scenes":[{"scene_title":"하나","original_prompt":"wrong","character_ids":["robot"],"scene_prompt_en":"Sol stands beside a lake.","strategy":"storyboard","change_summary":"하나"},{"scene_title":"둘","original_prompt":"wrong","character_ids":["robot"],"scene_prompt_en":"Sol walks into a forest.","strategy":"storyboard","change_summary":"둘"}]}`
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": content}}}})
	}))
	defer promptEngine.Close()
	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"prompt": {Endpoint: promptEngine.URL}}, PromptEnhancement: config.PromptEnhancement{Model: "planner"}}
	store, _ := jobs.New(cfg.DataDir)
	res := httptest.NewRecorder()
	const bible = "노란 사각 얼굴 로봇과 수채화 숲을 유지한다."
	req := httptest.NewRequest(http.MethodPost, "/api/prompts/sequence-plan", bytes.NewBufferString(`{"prompts":["호숫가에 선다","숲으로 걷는다"],"shared_prompt":"`+bible+`"}`))
	req.Header.Set("Content-Type", "application/json")
	New(cfg, store, nil).Handler().ServeHTTP(res, req)
	if res.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	var plan imageSequencePromptPlan
	_ = json.Unmarshal(res.Body.Bytes(), &plan)
	if plan.SharedPrompt != bible {
		t.Fatalf("edited bible changed: %q", plan.SharedPrompt)
	}
}

func TestStoryboardPlannerPreservesLockedVisualCharacterOnlyInSelectedScenes(t *testing.T) {
	promptEngine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if json.NewDecoder(r.Body).Decode(&request) != nil || len(request.Messages) != 2 || !strings.Contains(request.Messages[1].Content, `"locked_characters"`) || !strings.Contains(request.Messages[1].Content, "three turquoise wheel hubs") {
			t.Fatalf("locked character was not provided to planner: %#v", request)
		}
		// The model deliberately rewrites the identity, while scene two deliberately
		// excludes this character. Identity text must stay verbatim without adding an
		// unwanted person to scene two.
		content := `{"shared_prompt":"고정 로봇과 미래 도시","world_prompt_en":"A restrained cinematic future city.","characters":[{"id":"character_1","name_ko":"잘못 바꿈","name_en":"Wrong","description_ko":"잘못 바꿈","prompt_en":"A generic robot."}],"scenes":[{"scene_title":"출발","original_prompt":"wrong","character_ids":["character_1"],"scene_prompt_en":"Rho leaves a repair bay.","strategy":"storyboard","change_summary":"출발"},{"scene_title":"도착","original_prompt":"wrong","character_ids":[],"scene_prompt_en":"Rho reaches a rooftop garden.","strategy":"storyboard","change_summary":"도착"}]}`
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": content}}}})
	}))
	defer promptEngine.Close()
	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"prompt": {Endpoint: promptEngine.URL}}, PromptEnhancement: config.PromptEnhancement{Model: "planner"}}
	store, _ := jobs.New(cfg.DataDir)
	const locked = "Rho is a compact orange delivery robot with a cream circular faceplate, two cobalt eyes, and three turquoise wheel hubs on each side."
	body := bytes.NewBufferString(`{"prompts":["수리점에서 출발","옥상에 도착"],"locked_characters":[{"id":"character_1","name_ko":"로","name_en":"Rho","description_ko":"주황색 배달 로봇","prompt_en":"` + locked + `"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/prompts/sequence-plan", body)
	req.Header.Set("Content-Type", "application/json")
	res := httptest.NewRecorder()
	New(cfg, store, nil).Handler().ServeHTTP(res, req)
	if res.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	var plan imageSequencePromptPlan
	_ = json.Unmarshal(res.Body.Bytes(), &plan)
	if plan.Characters[0].PromptEN != locked || plan.Characters[0].NameEN != "Rho" {
		t.Fatalf("locked identity was rewritten: %#v", plan.Characters[0])
	}
	if strings.Count(plan.Scenes[0].EnhancedPrompt, locked) != 1 {
		t.Fatalf("locked prompt must appear once in selected scene: %q", plan.Scenes[0].EnhancedPrompt)
	}
	if strings.Contains(plan.Scenes[1].EnhancedPrompt, locked) {
		t.Fatalf("locked prompt leaked into a scene where the character is absent: %q", plan.Scenes[1].EnhancedPrompt)
	}
}
