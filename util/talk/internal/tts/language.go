package tts

import (
	"strings"
	"unicode"

	"github.com/abadojack/whatlanggo"
)

type SpeechPart struct {
	Text     string
	Language string
}

type textScript uint8

const (
	scriptUnknown textScript = iota
	scriptLatin
	scriptHangul
	scriptCJK
	scriptArabic
	scriptDevanagari
)

var magpieLatinLanguages = map[whatlanggo.Lang]string{
	whatlanggo.Eng: "en-US",
	whatlanggo.Spa: "es-ES",
	whatlanggo.Deu: "de-DE",
	whatlanggo.Fra: "fr-FR",
	whatlanggo.Ita: "it-IT",
	whatlanggo.Por: "pt-BR",
	whatlanggo.Vie: "vi-VN",
}

var magpieLatinWhitelist = func() map[whatlanggo.Lang]bool {
	result := make(map[whatlanggo.Lang]bool, len(magpieLatinLanguages))
	for language := range magpieLatinLanguages {
		result[language] = true
	}
	return result
}()

// SpeechParts resolves SparkTalk's auto language into concrete Magpie locale
// codes. The Magpie API itself accepts exactly one language per request, so
// mixed-script replies are synthesized as consecutive raw-PCM parts.
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
		return []SpeechPart{{Text: text, Language: configured}}
	}
	return splitMagpieLanguages(text, c.cfg.HanjaReading)
}

func splitMagpieLanguages(text, hanjaReading string) []SpeechPart {
	runes := []rune(text)
	if len(runes) == 0 {
		return nil
	}
	current := scriptUnknown
	start := 0
	sentenceBoundary := -1
	parts := make([]SpeechPart, 0, 4)
	lastTrailingSpace := false
	appendPart := func(end int, kind textScript) {
		rawValue := string(runes[start:end])
		leadingSpace := len(rawValue) > 0 && unicode.IsSpace([]rune(rawValue)[0])
		rawRunes := []rune(rawValue)
		trailingSpace := len(rawRunes) > 0 && unicode.IsSpace(rawRunes[len(rawRunes)-1])
		value := strings.TrimSpace(rawValue)
		if value == "" {
			return
		}
		language := languageForScript(value, kind)
		if kind == scriptCJK && language == "zh-CN" {
			switch hanjaReading {
			case "japanese":
				language = "ja-JP"
			case "chinese":
				// Keep the original Han characters and Mandarin locale.
			default:
				value, language = koreanizeHanja(value), "ko-KR"
			}
		}
		if len(parts) > 0 && parts[len(parts)-1].Language == language {
			separator := ""
			if lastTrailingSpace || leadingSpace {
				separator = " "
			}
			parts[len(parts)-1].Text += separator + value
		} else {
			parts = append(parts, SpeechPart{Text: value, Language: language})
		}
		lastTrailingSpace = trailingSpace
	}
	for index, character := range runes {
		if isSentenceBoundary(character) {
			sentenceBoundary = index + 1
		}
		next := scriptOf(character)
		if next == scriptUnknown {
			continue
		}
		if current == scriptUnknown {
			current = next
			continue
		}
		if sentenceBoundary > start && index >= sentenceBoundary {
			appendPart(sentenceBoundary, current)
			start = sentenceBoundary
			current = next
			sentenceBoundary = -1
			continue
		}
		if next == current {
			continue
		}
		appendPart(index, current)
		start = index
		current = next
		sentenceBoundary = -1
	}
	if current == scriptUnknown {
		// Numbers and symbols carry no language evidence. SparkTalk's primary UI
		// language is Korean, so its text normalizer is the safest default.
		return []SpeechPart{{Text: text, Language: "ko-KR"}}
	}
	appendPart(len(runes), current)
	return parts
}

func isSentenceBoundary(character rune) bool {
	switch character {
	case '.', '!', '?', '\n', '。', '！', '？':
		return true
	default:
		return false
	}
}

func scriptOf(character rune) textScript {
	switch {
	case unicode.Is(unicode.Hangul, character):
		return scriptHangul
	case unicode.Is(unicode.Hiragana, character), unicode.Is(unicode.Katakana, character), unicode.Is(unicode.Han, character):
		return scriptCJK
	case unicode.Is(unicode.Arabic, character):
		return scriptArabic
	case unicode.Is(unicode.Devanagari, character):
		return scriptDevanagari
	case unicode.Is(unicode.Latin, character):
		return scriptLatin
	default:
		return scriptUnknown
	}
}

func languageForScript(text string, kind textScript) string {
	switch kind {
	case scriptHangul:
		return "ko-KR"
	case scriptCJK:
		for _, character := range text {
			if unicode.Is(unicode.Hiragana, character) || unicode.Is(unicode.Katakana, character) {
				return "ja-JP"
			}
		}
		return "zh-CN"
	case scriptArabic:
		return "ar-MSA"
	case scriptDevanagari:
		return "hi-IN"
	case scriptLatin:
		return detectMagpieLatinLanguage(text)
	default:
		return "en-US"
	}
}

func detectMagpieLatinLanguage(text string) string {
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
	info := whatlanggo.DetectWithOptions(text, whatlanggo.Options{Whitelist: magpieLatinWhitelist})
	if language, ok := magpieLatinLanguages[info.Lang]; ok && (info.IsReliable() || letters >= 12) {
		return language
	}
	return "en-US"
}
