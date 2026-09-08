package tts

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"sparktalk/internal/config"
)

func TestSpeechUsesMinimalMagpiePayload(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if request["model"] != "magpietts" || request["language"] != "ko-KR" || request["voice"] != "Sofia" {
			t.Fatalf("unexpected Magpie request: %+v", request)
		}
		for _, unsupported := range []string{"seed", "instructions", "task_type", "stream_format"} {
			if _, ok := request[unsupported]; ok {
				t.Fatalf("Magpie request contains %q: %+v", unsupported, request)
			}
		}
		w.Header().Set("Content-Type", "audio/pcm")
		_, _ = w.Write([]byte("pcm"))
	}))
	defer server.Close()

	client := New(config.TTSConfig{Enabled: true, Endpoint: server.URL, Model: "magpietts", Language: "ko-KR", Voice: "Sofia", SampleRate: 22050, Timeout: "5s"})
	stream, err := client.SpeechStream(context.Background(), "안녕하세요")
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Body.Close()
	if stream.SampleRate != 22050 {
		t.Fatalf("sample rate = %d, want 22050", stream.SampleRate)
	}
}

func TestMagpieSpeechPartsSwitchLatinRunsToEnglish(t *testing.T) {
	client := New(config.TTSConfig{Language: "auto", HanjaReading: "korean"})
	parts := client.SpeechParts("한국어 API 테스트와 English sentence입니다.")
	want := []SpeechPart{
		{Text: "한국어", Language: "ko-KR"},
		{Text: "API", Language: "en-US"},
		{Text: "테스트와", Language: "ko-KR"},
		{Text: "English sentence", Language: "en-US"},
		{Text: "입니다.", Language: "ko-KR"},
	}
	if !reflect.DeepEqual(parts, want) {
		t.Fatalf("unexpected mixed-language parts: %#v", parts)
	}
}

func TestMagpieSpeechPartsDetectSupportedScripts(t *testing.T) {
	client := New(config.TTSConfig{Language: "auto", HanjaReading: "chinese"})
	parts := client.SpeechParts("안녕하세요. 今日は晴れです。 今天天气很好。 مرحبا. नमस्ते.")
	want := []SpeechPart{
		{Text: "안녕하세요.", Language: "ko-KR"},
		{Text: "今日は晴れです。", Language: "ja-JP"},
		{Text: "今天天气很好。", Language: "zh-CN"},
		{Text: "مرحبا.", Language: "ar-MSA"},
		{Text: "नमस्ते.", Language: "hi-IN"},
	}
	if !reflect.DeepEqual(parts, want) {
		t.Fatalf("unexpected script language parts: %#v", parts)
	}
}

func TestMagpieSpeechPartsDetectLatinLanguages(t *testing.T) {
	client := New(config.TTSConfig{Language: "auto", HanjaReading: "korean"})
	tests := map[string]string{
		"Where there is a will there is a way.":                           "en-US",
		"Además de todo lo anterior, esta frase está escrita en español.": "es-ES",
		"Dies ist ein deutscher Satz mit mehreren eindeutigen Wörtern.":   "de-DE",
		"Ceci est une phrase française avec plusieurs mots distinctifs.":  "fr-FR",
		"Questa è una frase italiana con diverse parole riconoscibili.":   "it-IT",
		"Esta é uma frase em português com várias palavras conhecidas.":   "pt-BR",
		"Đây là một câu tiếng Việt có nhiều từ dễ nhận biết.":             "vi-VN",
	}
	for text, want := range tests {
		parts := client.SpeechParts(text)
		if len(parts) != 1 || parts[0].Language != want {
			t.Errorf("%q language = %#v, want %s", text, parts, want)
		}
	}
}

func TestMagpieExplicitLanguageIsNotOverridden(t *testing.T) {
	client := New(config.TTSConfig{Language: "ja-JP"})
	parts := client.SpeechParts("API テスト")
	want := []SpeechPart{{Text: "API テスト", Language: "ja-JP"}}
	if !reflect.DeepEqual(parts, want) {
		t.Fatalf("explicit language was overridden: %#v", parts)
	}
}

func TestMagpieSpeechPartsUseKoreanForLanguageNeutralText(t *testing.T) {
	client := New(config.TTSConfig{Language: "auto", HanjaReading: "korean"})
	want := []SpeechPart{{Text: "123%", Language: "ko-KR"}}
	if parts := client.SpeechParts("123%"); !reflect.DeepEqual(parts, want) {
		t.Fatalf("neutral text parts = %#v, want %#v", parts, want)
	}
}

func TestMagpieSpeechPartsReadHanjaInKorean(t *testing.T) {
	client := New(config.TTSConfig{Language: "auto", HanjaReading: "korean"})
	parts := client.SpeechParts("大韓民國은 民主共和國이다. 女子와 李氏")
	want := []SpeechPart{{Text: "대한민국은 민주공화국이다. 여자와 이씨", Language: "ko-KR"}}
	if !reflect.DeepEqual(parts, want) {
		t.Fatalf("Korean Hanja parts = %#v, want %#v", parts, want)
	}
}

func TestMagpieSpeechPartsReadKanjiInJapanese(t *testing.T) {
	client := New(config.TTSConfig{Language: "auto", HanjaReading: "japanese"})
	parts := client.SpeechParts("日本國 東京都 世界平和")
	want := []SpeechPart{{Text: "日本國 東京都 世界平和", Language: "ja-JP"}}
	if !reflect.DeepEqual(parts, want) {
		t.Fatalf("Japanese Kanji parts = %#v, want %#v", parts, want)
	}
}

func TestMagpieExplicitKoreanConvertsHanja(t *testing.T) {
	client := New(config.TTSConfig{Language: "ko-KR", HanjaReading: "chinese"})
	want := []SpeechPart{{Text: "대한민국", Language: "ko-KR"}}
	if parts := client.SpeechParts("大韓民國"); !reflect.DeepEqual(parts, want) {
		t.Fatalf("explicit Korean Hanja parts = %#v, want %#v", parts, want)
	}
}
