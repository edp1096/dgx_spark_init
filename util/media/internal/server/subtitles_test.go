package server

import (
	"strings"
	"testing"
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
