package tts

import (
	"strings"
	"unicode"

	"github.com/abadojack/whatlanggo"
)

type SpeechPart struct {
	Text     string `json:"text"`
	Language string `json:"language"`
}

var qwenLatinLanguages = map[whatlanggo.Lang]string{
	whatlanggo.Eng: "en-US",
	whatlanggo.Spa: "es-ES",
	whatlanggo.Deu: "de-DE",
	whatlanggo.Fra: "fr-FR",
	whatlanggo.Ita: "it-IT",
	whatlanggo.Por: "pt-BR",
}

var qwenLatinWhitelist = func() map[whatlanggo.Lang]bool {
	result := make(map[whatlanggo.Lang]bool, len(qwenLatinLanguages))
	for language := range qwenLatinLanguages {
		result[language] = true
	}
	return result
}()

// SpeechParts selects one language per sentence. Keep English names and
// acronyms inside Korean/Japanese/Chinese sentences for the multilingual model.
func (c *Client) SpeechParts(text string) []SpeechPart {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	configured := strings.TrimSpace(c.cfg.Language)
	if !strings.EqualFold(configured, "auto") {
		if strings.HasPrefix(strings.ToLower(configured), "ko") {
			text = koreanizeHanja(text)
		}
		return boundedSpeechParts([]SpeechPart{{Text: text, Language: configured}})
	}
	return boundedSpeechParts(splitQwenLanguages(text, c.cfg.HanjaReading))
}

// Bound manual full-reply requests as well as automatically streamed replies.
// The native talker has a finite text/audio context; prefer sentence/word cuts
// so a long reply doesn't become one oversized synthesis or split every acronym.
const maxSpeechPartRunes = 384

func boundedSpeechParts(parts []SpeechPart) []SpeechPart {
	result := make([]SpeechPart, 0, len(parts))
	for _, part := range parts {
		runes := []rune(part.Text)
		for len(runes) > maxSpeechPartRunes {
			cut, sentence, word := maxSpeechPartRunes, 0, 0
			for i, r := range runes[:maxSpeechPartRunes] {
				if speechSentenceBoundary(runes, i) {
					sentence = i + 1
				}
				if unicode.IsSpace(r) {
					word = i + 1
				}
			}
			if sentence >= maxSpeechPartRunes/3 {
				cut = sentence
			} else if word >= maxSpeechPartRunes/2 {
				cut = word
			}
			value := strings.TrimSpace(string(runes[:cut]))
			if value != "" {
				result = append(result, SpeechPart{Text: value, Language: part.Language})
			}
			runes = []rune(strings.TrimSpace(string(runes[cut:])))
		}
		if value := strings.TrimSpace(string(runes)); value != "" {
			result = append(result, SpeechPart{Text: value, Language: part.Language})
		}
	}
	return result
}

func splitQwenLanguages(text, hanjaReading string) []SpeechPart {
	runes := []rune(text)
	parts := make([]SpeechPart, 0, 4)
	appendPart := func(value string) {
		value = strings.TrimSpace(value)
		if value == "" {
			return
		}
		hasHangul, hasKana, hasHan, hasLatin, hasCyrillic := false, false, false, false, false
		for _, r := range value {
			hasHangul = hasHangul || unicode.Is(unicode.Hangul, r)
			hasKana = hasKana || unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r)
			hasHan = hasHan || unicode.Is(unicode.Han, r)
			hasLatin = hasLatin || unicode.Is(unicode.Latin, r)
			hasCyrillic = hasCyrillic || unicode.Is(unicode.Cyrillic, r)
		}
		language := "ko-KR"
		switch {
		case hasHangul:
			value = koreanizeHanja(value)
		case hasKana:
			language = "ja-JP"
		case hasHan:
			switch hanjaReading {
			case "japanese":
				language = "ja-JP"
			case "chinese":
				language = "zh-CN"
			default:
				value = koreanizeHanja(value)
			}
		case hasCyrillic:
			language = "ru-RU"
		case hasLatin:
			language = detectQwenLatinLanguage(value)
		}
		if len(parts) > 0 && parts[len(parts)-1].Language == language {
			parts[len(parts)-1].Text += " " + value
		} else {
			parts = append(parts, SpeechPart{Text: value, Language: language})
		}
	}
	start := 0
	for i := range runes {
		if !speechSentenceBoundary(runes, i) {
			continue
		}
		appendPart(string(runes[start : i+1]))
		start = i + 1
	}
	appendPart(string(runes[start:]))
	return parts
}

func speechSentenceBoundary(runes []rune, i int) bool {
	r := runes[i]
	if !isSentenceBoundary(r) {
		return false
	}
	if r == '.' && i > 0 && i+1 < len(runes) {
		prev, next := runes[i-1], runes[i+1]
		if (unicode.IsDigit(prev) && unicode.IsDigit(next)) ||
			(unicode.Is(unicode.Latin, prev) && unicode.Is(unicode.Latin, next)) {
			return false
		}
	}
	return true
}

func isSentenceBoundary(character rune) bool {
	switch character {
	case '.', '!', '?', '\n', '。', '！', '？':
		return true
	default:
		return false
	}
}

func detectQwenLatinLanguage(text string) string {
	letters := 0
	asciiOnly := true
	for _, character := range text {
		if unicode.IsLetter(character) {
			letters++
			if character > unicode.MaxASCII {
				asciiOnly = false
			}
		}
	}
	// Acronyms and product names do not contain enough evidence for statistical
	// identification. Short ASCII fragments embedded in another script are most
	// commonly English; users can select an explicit locale for ambiguous text.
	if letters < 4 || (asciiOnly && letters < 20) {
		return "en-US"
	}
	info := whatlanggo.DetectWithOptions(text, whatlanggo.Options{Whitelist: qwenLatinWhitelist})
	if language, ok := qwenLatinLanguages[info.Lang]; ok && (info.IsReliable() || letters >= 12) {
		return language
	}
	return "en-US"
}
