package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
)

func TestSubtitleRenderersPreserveTimingAndBilingualText(t *testing.T) {
	segments := []subtitleCue{
		{Start: 0, End: 1.25, Text: "Hello", Translated: "안녕하세요"},
		{Start: 1.25, End: 65.5, Text: "World", Translated: "세계"},
	}
	srt := renderSRT(segments, "bilingual")
	if !strings.Contains(srt, "00:00:00,000 --> 00:00:01,250") || !strings.Contains(srt, "Hello\n안녕하세요") {
		t.Fatalf("unexpected SRT:\n%s", srt)
	}
	vtt := renderVTT(segments, "translated")
	if !strings.HasPrefix(vtt, "WEBVTT\n\n") || !strings.Contains(vtt, "00:00:01.250 --> 00:01:05.500\n세계") {
		t.Fatalf("unexpected VTT:\n%s", vtt)
	}
	timestamped := renderTimestampedText(segments, "none")
	if !strings.Contains(timestamped, "[00:00:01.250 --> 00:01:05.500] World") {
		t.Fatalf("unexpected timestamped text:\n%s", timestamped)
	}
}

func TestParseRenderedBilingualSubtitleForRegeneration(t *testing.T) {
	srt := `1
00:00:00,160 --> 00:00:05,600
A great sports movie transcends its limits.
훌륭한 스포츠 영화는 한계를 뛰어넘습니다.

2
00:00:05,920 --> 00:00:06,560
시장은 직접 확인했지.
시장은 직접 확인했지.`
	cues, err := parseRenderedSubtitle(srt, "srt", "bilingual", "Korean")
	if err != nil || len(cues) != 2 {
		t.Fatalf("parse failed: cues=%#v err=%v", cues, err)
	}
	if cues[0].Text != "A great sports movie transcends its limits." || cues[0].Translated != "훌륭한 스포츠 영화는 한계를 뛰어넘습니다." {
		t.Fatalf("unexpected translated cue: %#v", cues[0])
	}
	if cues[1].Text != "시장은 직접 확인했지." || cues[1].Translated != "시장은 직접 확인했지." {
		t.Fatalf("duplicated Korean cue was not split: %#v", cues[1])
	}
	if got := renderSRT(cues, "none"); strings.Count(got, "시장은 직접 확인했지.") != 1 {
		t.Fatalf("original-only regeneration duplicated text:\n%s", got)
	}
}

func TestRegenerateSubtitleReusesExistingBilingualCues(t *testing.T) {
	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	const id = "subtitle-regenerate-test"
	srt := "1\n00:00:00,000 --> 00:00:01,000\nHello\n안녕하세요\n"
	if err := os.WriteFile(store.OutputPath(id+".srt"), []byte(srt), 0o644); err != nil {
		t.Fatal(err)
	}
	job := jobs.Job{
		ID: id, Kind: "recognition", Status: "completed", Prompt: "example.mp4", CreatedAt: time.Now(),
		Params:  map[string]any{"translation_mode": "bilingual", "target_language": "Korean", "output_formats": []string{"srt"}},
		Outputs: map[string]string{"srt": "/api/outputs/" + id + ".srt"},
	}
	if err := store.Save(job); err != nil {
		t.Fatal(err)
	}
	handler := New(config.Config{DataDir: dataDir}, store, nil).Handler()
	body, _ := json.Marshal(subtitleRegenerateRequest{TranslationMode: "none", OutputFormats: []string{"srt", "txt"}})
	request := httptest.NewRequest(http.MethodPost, "/api/jobs/"+id+"/subtitle-regenerate", bytes.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	regenerated, err := os.ReadFile(store.OutputPath(id + ".srt"))
	if err != nil || strings.Contains(string(regenerated), "안녕하세요") || !strings.Contains(string(regenerated), "Hello") {
		t.Fatalf("unexpected regenerated SRT=%q err=%v", regenerated, err)
	}
	if _, err := os.Stat(store.OutputPath(id + ".cues.json")); err != nil {
		t.Fatalf("cue archive was not migrated: %v", err)
	}
	updated, _ := store.Get(id)
	if updated.Params["translation_mode"] != "none" || updated.Outputs["txt"] == "" {
		t.Fatalf("job was not updated: %#v", updated)
	}
}

func TestParseMarkedTranslations(t *testing.T) {
	parsed := parseMarkedTranslations("[[0002]] 첫 줄\n이어지는 줄\n[[0003]] 다음 줄")
	if parsed[2] != "첫 줄 이어지는 줄" || parsed[3] != "다음 줄" {
		t.Fatalf("unexpected translations: %#v", parsed)
	}
}

func TestCuesFromTimestampsRestorePunctuationAndOffset(t *testing.T) {
	words := []timedWord{
		{Text: "안녕", Start: 0.1, End: 0.5},
		{Text: "반가워", Start: 0.6, End: 1.2},
		{Text: "다음", Start: 1.5, End: 1.8},
		{Text: "문장", Start: 1.9, End: 2.2},
	}
	cues := cuesFromTimestamps("안녕, 반가워! 다음 문장.", words, 180)
	if len(cues) != 2 {
		t.Fatalf("expected 2 cues, got %#v", cues)
	}
	if cues[0].Text != "안녕, 반가워!" || cues[0].Start != 180.1 || cues[0].End != 181.2 {
		t.Fatalf("unexpected first cue: %#v", cues[0])
	}
	if cues[1].Text != "다음 문장." || cues[1].Start != 181.5 || cues[1].End != 182.2 {
		t.Fatalf("unexpected second cue: %#v", cues[1])
	}
}

func TestValidateAlignedResultRejectsHallucinatedTimingAndRepetition(t *testing.T) {
	if err := validateAlignedResult("정상 문장입니다.", []timedWord{{Text: "정상", Start: 0.2, End: 1.0}}, 2, false); err != nil {
		t.Fatalf("valid result rejected: %v", err)
	}
	if err := validateAlignedResult("bad", []timedWord{{Text: "bad", Start: 0, End: 88}}, 30, false); err == nil {
		t.Fatal("out-of-range timestamp was accepted")
	}
	repeated := strings.Repeat("You're the one who's been lying to me. ", 6)
	if err := validateAlignedResult(repeated, []timedWord{{Text: "You're", Start: 0, End: 1}}, 30, false); err == nil {
		t.Fatal("repeated hallucination was accepted")
	}
	if err := validateAlignedResult(repeated, []timedWord{{Text: "You're", Start: 0, End: 1}}, 30, true); err != nil {
		t.Fatalf("repeated song lyrics were rejected in multilingual mode: %v", err)
	}
	collapsed := make([]timedWord, 20)
	for index := range collapsed {
		collapsed[index] = timedWord{Text: "word", Start: 0, End: 0}
	}
	if err := validateAlignedResult("many aligned words", collapsed, 30, true); err == nil {
		t.Fatal("collapsed timestamps were accepted")
	}
}

func TestAutomaticLanguageModesAndDetectedLanguageMerge(t *testing.T) {
	if !isSingleLanguageAuto("Auto") || !isAutomaticLanguage("auto") {
		t.Fatal("single-language auto mode was not recognized")
	}
	if !isMultilingualAuto("AutoMultilingual") || !isAutomaticLanguage("automultilingual") {
		t.Fatal("multilingual auto mode was not recognized")
	}
	if got := mergeDetectedLanguages("Japanese,English", "English, Korean"); got != "Japanese,English,Korean" {
		t.Fatalf("unexpected detected-language merge: %q", got)
	}
}

func TestValidSubtitleTranslationDetectsUntranslatedKoreanTarget(t *testing.T) {
	if validSubtitleTranslation("状態不明です。", "状態不明です。", "Korean") {
		t.Fatal("unchanged Japanese was accepted as Korean")
	}
	if !validSubtitleTranslation("状態不明です。", "상태를 알 수 없습니다.", "Korean") {
		t.Fatal("Korean translation was rejected")
	}
}

func TestTranslateSubtitleSegmentsKeepsSourceAndContinuesAfterInvalidCue(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"content": "状態不明です。"}}},
		})
	}))
	defer engine.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	service := New(config.Config{
		DataDir:           dataDir,
		Engines:           map[string]config.Engine{"prompt": {Endpoint: engine.URL}},
		PromptEnhancement: config.PromptEnhancement{Model: "test-translator"},
	}, store, nil)
	cues := []subtitleCue{{Start: 0, End: 1, Text: "状態不明です。"}}
	progress := 0
	warnings, err := service.translateSubtitleSegments(cues, "Korean", func(done, _ int) { progress = done }, nil)
	if err != nil {
		t.Fatalf("invalid individual translation stopped the job: %v", err)
	}
	if progress != 1 || len(warnings) != 1 {
		t.Fatalf("progress=%d warnings=%#v", progress, warnings)
	}
	if warnings[0].Segment != 1 || warnings[0].Source != cues[0].Text || warnings[0].Reason == "" {
		t.Fatalf("unexpected translation warning: %#v", warnings[0])
	}
	if cues[0].Translated != cues[0].Text {
		t.Fatalf("fallback=%q, want original %q", cues[0].Translated, cues[0].Text)
	}
}

func TestTranslateSubtitleSegmentsDoesNotHideEngineFailure(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "offline", http.StatusServiceUnavailable)
	}))
	defer engine.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	service := New(config.Config{
		DataDir:           dataDir,
		Engines:           map[string]config.Engine{"prompt": {Endpoint: engine.URL}},
		PromptEnhancement: config.PromptEnhancement{Model: "test-translator"},
	}, store, nil)
	cues := []subtitleCue{{Start: 0, End: 1, Text: "状態不明です。"}}
	warnings, err := service.translateSubtitleSegments(cues, "Korean", nil, nil)
	if err == nil {
		t.Fatal("translation engine outage was incorrectly treated as a recoverable cue warning")
	}
	if len(warnings) != 0 || cues[0].Translated != "" {
		t.Fatalf("warnings=%#v translated=%q", warnings, cues[0].Translated)
	}
}
