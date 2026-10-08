package tts

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"sparktalk/internal/config"
)

func TestSpeechUsesMinimalQwenPayload(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if request["model"] != "qwen3-tts-0.6b-q8" || request["language"] != "Korean" || request["voice"] != "sohee" {
			t.Fatalf("unexpected Qwen request: %+v", request)
		}
		if request["seed"] != float64(42) {
			t.Fatalf("sampling differs from tested preset: %+v", request)
		}
		for _, unsupported := range []string{"instructions", "task_type", "stream_format"} {
			if _, ok := request[unsupported]; ok {
				t.Fatalf("Qwen request contains %q: %+v", unsupported, request)
			}
		}
		w.Header().Set("Content-Type", "audio/pcm")
		_, _ = w.Write([]byte("pcm"))
	}))
	defer server.Close()

	client := New(config.TTSConfig{Enabled: true, Endpoint: server.URL, Model: "qwen3-tts-0.6b-q8", Language: "ko-KR", Voice: "sohee", SampleRate: 24000, Timeout: "5s"})
	stream, err := client.SpeechStream(context.Background(), "안녕하세요")
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Body.Close()
	if stream.SampleRate != 24000 {
		t.Fatalf("sample rate = %d, want 24000", stream.SampleRate)
	}
}

func TestQwenLanguageLocalesAndUnsupportedLanguage(t *testing.T) {
	for locale, want := range map[string]string{"ko-KR": "Korean", "ja-JP": "Japanese", "zh-CN": "Chinese", "en-US": "English", "ru-RU": "Russian", "German": "German", "pt-BR": "Portuguese"} {
		got, err := qwenLanguage(locale)
		if err != nil || got != want {
			t.Errorf("%s: %s %v", locale, got, err)
		}
	}
	if _, err := qwenLanguage("ar-MSA"); err == nil {
		t.Fatal("unsupported language accepted")
	}
}

func TestQwenSpeechPartsPreserveBrandsAndDecimalNumbers(t *testing.T) {
	client := New(config.TTSConfig{Language: "auto", HanjaReading: "korean"})
	text := "Nemotron Speech와 함께 경험해 보세요. Qwen3.8 API는 3.14초입니다."
	want := []SpeechPart{{Text: text, Language: "ko-KR"}}
	if got := client.SpeechParts(text); !reflect.DeepEqual(got, want) {
		t.Fatal("name or decimal fragmented", got)
	}
}

func TestQwenLongRepliesAreBoundedWithoutLosingText(t *testing.T) {
	text := strings.Repeat("한국어 API와 Nemotron Speech를 함께 읽습니다. ", 100)
	for _, language := range []string{"auto", "ko-KR"} {
		client := New(config.TTSConfig{Language: language, HanjaReading: "korean"})
		parts := client.SpeechParts(text)
		if len(parts) < 2 {
			t.Fatal("long request remained unbounded")
		}
		var read []string
		for _, part := range parts {
			if len([]rune(part.Text)) > maxSpeechPartRunes || part.Language != "ko-KR" {
				t.Fatal("oversized or misclassified part", part)
			}
			read = append(read, part.Text)
		}
		if strings.Join(strings.Fields(strings.Join(read, " ")), " ") != strings.Join(strings.Fields(text), " ") {
			t.Fatal("reply content lost at chunk boundaries")
		}
	}
}

func TestQwenLongReplyDoesNotCutDottedModelName(t *testing.T) {
	client := New(config.TTSConfig{Language: "ko-KR"})
	text := strings.Repeat("가", 370) + " Qwen3.8 " + strings.Repeat("나", 370)
	parts := client.SpeechParts(text)
	wholeName := false
	for _, part := range parts {
		wholeName = wholeName || strings.Contains(part.Text, "Qwen3.8")
	}
	if !wholeName {
		t.Fatal("model name split across synthesis calls", parts)
	}
}

func TestQwenSpeechPartsKeepMixedSentenceTogether(t *testing.T) {
	client := New(config.TTSConfig{Language: "auto", HanjaReading: "korean"})
	text := "한국어 API 테스트와 English sentence입니다."
	want := []SpeechPart{{Text: text, Language: "ko-KR"}}
	if parts := client.SpeechParts(text); !reflect.DeepEqual(parts, want) {
		t.Fatalf("mixed sentence fragmented: %#v", parts)
	}
}

func TestQwenSpeechPartsDetectSupportedScripts(t *testing.T) {
	client := New(config.TTSConfig{Language: "auto", HanjaReading: "chinese"})
	parts := client.SpeechParts("안녕하세요. 今日は晴れです。 今天天气很好。 Это русский текст.")
	want := []SpeechPart{
		{Text: "안녕하세요.", Language: "ko-KR"},
		{Text: "今日は晴れです。", Language: "ja-JP"},
		{Text: "今天天气很好。", Language: "zh-CN"},
		{Text: "Это русский текст.", Language: "ru-RU"},
	}
	if !reflect.DeepEqual(parts, want) {
		t.Fatalf("unexpected script language parts: %#v", parts)
	}
}

func TestQwenSpeechPartsDetectLatinLanguages(t *testing.T) {
	client := New(config.TTSConfig{Language: "auto", HanjaReading: "korean"})
	tests := map[string]string{
		"Where there is a will there is a way.":                           "en-US",
		"Además de todo lo anterior, esta frase está escrita en español.": "es-ES",
		"Dies ist ein deutscher Satz mit mehreren eindeutigen Wörtern.":   "de-DE",
		"Ceci est une phrase française avec plusieurs mots distinctifs.":  "fr-FR",
		"Questa è una frase italiana con diverse parole riconoscibili.":   "it-IT",
		"Esta é uma frase em português com várias palavras conhecidas.":   "pt-BR",
	}
	for text, want := range tests {
		parts := client.SpeechParts(text)
		if len(parts) != 1 || parts[0].Language != want {
			t.Errorf("%q language = %#v, want %s", text, parts, want)
		}
	}
}

func TestQwenExplicitLanguageIsNotOverridden(t *testing.T) {
	client := New(config.TTSConfig{Language: "ja-JP"})
	parts := client.SpeechParts("API テスト")
	want := []SpeechPart{{Text: "API テスト", Language: "ja-JP"}}
	if !reflect.DeepEqual(parts, want) {
		t.Fatalf("explicit language was overridden: %#v", parts)
	}
}

func TestQwenSpeechPartsUseKoreanForLanguageNeutralText(t *testing.T) {
	client := New(config.TTSConfig{Language: "auto", HanjaReading: "korean"})
	want := []SpeechPart{{Text: "123%", Language: "ko-KR"}}
	if parts := client.SpeechParts("123%"); !reflect.DeepEqual(parts, want) {
		t.Fatalf("neutral text parts = %#v, want %#v", parts, want)
	}
}

func TestQwenSpeechPartsReadHanjaInKorean(t *testing.T) {
	client := New(config.TTSConfig{Language: "auto", HanjaReading: "korean"})
	parts := client.SpeechParts("大韓民國은 民主共和國이다. 女子와 李氏")
	want := []SpeechPart{{Text: "대한민국은 민주공화국이다. 여자와 이씨", Language: "ko-KR"}}
	if !reflect.DeepEqual(parts, want) {
		t.Fatalf("Korean Hanja parts = %#v, want %#v", parts, want)
	}
}

func TestQwenSpeechPartsReadKanjiInJapanese(t *testing.T) {
	client := New(config.TTSConfig{Language: "auto", HanjaReading: "japanese"})
	parts := client.SpeechParts("日本國 東京都 世界平和")
	want := []SpeechPart{{Text: "日本國 東京都 世界平和", Language: "ja-JP"}}
	if !reflect.DeepEqual(parts, want) {
		t.Fatalf("Japanese Kanji parts = %#v, want %#v", parts, want)
	}
}

func TestQwenExplicitKoreanConvertsHanja(t *testing.T) {
	client := New(config.TTSConfig{Language: "ko-KR", HanjaReading: "chinese"})
	want := []SpeechPart{{Text: "대한민국", Language: "ko-KR"}}
	if parts := client.SpeechParts("大韓民國"); !reflect.DeepEqual(parts, want) {
		t.Fatalf("explicit Korean Hanja parts = %#v, want %#v", parts, want)
	}
}
