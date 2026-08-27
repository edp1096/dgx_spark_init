package server

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode/utf8"
)

const autoMultilingualLanguage = "AutoMultilingual"

type timedWord struct {
	Text  string  `json:"text"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

type subtitleCue struct {
	Start      float64 `json:"start"`
	End        float64 `json:"end"`
	Text       string  `json:"text"`
	Translated string  `json:"translated,omitempty"`
}

func (s *Server) recoverSubtitleSegment(inputDir, sourcePath string, absoluteOffset float64, language, context string) ([]subtitleCue, string, error) {
	var lastErr error
	for _, seconds := range []int{10, 5} {
		retryDir, err := os.MkdirTemp(inputDir, fmt.Sprintf("retry-%ds-", seconds))
		if err != nil {
			return nil, "", err
		}
		archivePath := filepath.Join(retryDir, "prepared.zip")
		fields := map[string]string{"segment_seconds": strconv.Itoa(seconds)}
		endpoint := s.config().Engines["media"].Endpoint + "/v1/media/prepare"
		err = s.callMultipartToFileStreaming(endpoint, fields, "file", []string{sourcePath}, archivePath)
		if err == nil {
			err = extractPreparedArchive(archivePath, filepath.Join(retryDir, "prepared"))
		}
		var manifest preparedManifest
		if err == nil {
			manifest, err = readPreparedManifest(filepath.Join(retryDir, "prepared"))
		}
		if err != nil {
			lastErr = err
			_ = os.RemoveAll(retryDir)
			continue
		}

		var cues []subtitleCue
		detectedLanguage := ""
		for _, segment := range manifest.Segments {
			text, detected, words, transcribeErr := s.transcribeSegment(
				filepath.Join(retryDir, "prepared", segment.Name), language, context,
			)
			if transcribeErr != nil {
				err = transcribeErr
				break
			}
			if validationErr := validateAlignedResult(text, words, segment.Duration, isMultilingualAuto(language)); validationErr != nil {
				err = validationErr
				break
			}
			if isMultilingualAuto(language) {
				detectedLanguage = mergeDetectedLanguages(detectedLanguage, detected)
			} else if detectedLanguage == "" && detected != "" {
				detectedLanguage = detected
			}
			segmentCues := cuesFromTimestamps(text, words, absoluteOffset+segment.Start)
			if len(segmentCues) == 0 && strings.TrimSpace(text) != "" {
				segmentCues = append(segmentCues, subtitleCue{
					Start: absoluteOffset + segment.Start,
					End:   absoluteOffset + segment.End,
					Text:  strings.TrimSpace(text),
				})
			}
			cues = append(cues, segmentCues...)
		}
		_ = os.RemoveAll(retryDir)
		if err == nil {
			return cues, detectedLanguage, nil
		}
		lastErr = err
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("split retry produced no result")
	}
	return nil, "", lastErr
}

func (s *Server) transcribeSegment(path, language, context string) (string, string, []timedWord, error) {
	cfg := s.config()
	fields := map[string]string{"model": cfg.Recognition.Model}
	if language != "" && !isAutomaticLanguage(language) {
		fields["language"] = language
	}
	if context != "" {
		fields["prompt"] = context
	}
	data, err := s.transcribeWithEngine(fields, path)
	if err != nil {
		return "", "", nil, err
	}
	var response struct {
		Text       string      `json:"text"`
		Language   string      `json:"language"`
		Timestamps []timedWord `json:"timestamps"`
	}
	if err := json.Unmarshal(data, &response); err != nil {
		return "", "", nil, err
	}
	return response.Text, response.Language, response.Timestamps, nil
}

func cuesFromTimestamps(transcript string, words []timedWord, offset float64) []subtitleCue {
	valid := make([]timedWord, 0, len(words))
	for _, word := range words {
		word.Text = strings.TrimSpace(word.Text)
		if word.Text != "" && word.End >= word.Start && word.Start >= 0 {
			valid = append(valid, word)
		}
	}
	if len(valid) == 0 {
		return nil
	}
	restored, exact := restoreAlignedText(strings.TrimSpace(transcript), valid)
	cues := make([]subtitleCue, 0, len(restored)/8+1)
	start := 0
	for index := range restored {
		text := cueTokenText(restored[start:index+1], exact)
		duration := restored[index].End - restored[start].Start
		if duration >= 6 || utf8.RuneCountInString(text) >= 60 || hasSentenceEnding(text) || index == len(restored)-1 {
			if text != "" {
				cues = append(cues, subtitleCue{
					Start: offset + restored[start].Start,
					End:   offset + restored[index].End,
					Text:  text,
				})
			}
			start = index + 1
		}
	}
	return cues
}

func validateAlignedResult(transcript string, words []timedWord, duration float64, allowRepeatedLyrics bool) error {
	if strings.TrimSpace(transcript) == "" || len(words) == 0 {
		return nil
	}
	outOfRange := 0
	for _, word := range words {
		if word.Start < -0.05 || word.End < word.Start || word.End > duration+0.5 {
			outOfRange++
		}
	}
	if outOfRange > 0 {
		return fmt.Errorf("aligner returned %d/%d timestamps outside %.3fs audio", outOfRange, len(words), duration)
	}
	if len(words) >= 12 {
		minimum, maximum := words[0].Start, words[0].End
		for _, word := range words[1:] {
			if word.Start < minimum {
				minimum = word.Start
			}
			if word.End > maximum {
				maximum = word.End
			}
		}
		if maximum-minimum < 0.25 {
			return fmt.Errorf("aligner collapsed %d words into %.3fs", len(words), maximum-minimum)
		}
	}
	if !allowRepeatedLyrics {
		for _, sentence := range strings.FieldsFunc(transcript, func(r rune) bool {
			return strings.ContainsRune(".!?。！？\n", r)
		}) {
			sentence = strings.TrimSpace(sentence)
			if utf8.RuneCountInString(sentence) >= 8 && strings.Count(transcript, sentence) >= 5 {
				return fmt.Errorf("ASR repeated the same sentence at least five times")
			}
		}
	}
	return nil
}

func isSingleLanguageAuto(language string) bool {
	return strings.EqualFold(strings.TrimSpace(language), "auto")
}

func isMultilingualAuto(language string) bool {
	return strings.EqualFold(strings.TrimSpace(language), autoMultilingualLanguage)
}

func isAutomaticLanguage(language string) bool {
	return isSingleLanguageAuto(language) || isMultilingualAuto(language)
}

func mergeDetectedLanguages(current, next string) string {
	seen := make(map[string]bool)
	merged := make([]string, 0, 4)
	for _, value := range []string{current, next} {
		for _, language := range strings.Split(value, ",") {
			language = strings.TrimSpace(language)
			key := strings.ToLower(language)
			if language == "" || seen[key] {
				continue
			}
			seen[key] = true
			merged = append(merged, language)
		}
	}
	return strings.Join(merged, ",")
}

// Forced Aligner는 문장부호를 제거한 어절을 반환한다. 원문에서 다음 어절까지의
// 공백과 문장부호를 앞 어절에 다시 붙여 자막 본문을 원래 표기대로 보존한다.
func restoreAlignedText(transcript string, words []timedWord) ([]timedWord, bool) {
	result := append([]timedWord(nil), words...)
	cursor := 0
	for index := range result {
		position := strings.Index(transcript[cursor:], result[index].Text)
		if position < 0 {
			return words, false
		}
		position += cursor
		if index == 0 {
			result[index].Text = transcript[:position] + result[index].Text
		} else {
			result[index-1].Text += transcript[cursor:position]
		}
		cursor = position + len(strings.TrimSpace(words[index].Text))
	}
	result[len(result)-1].Text += transcript[cursor:]
	return result, true
}

func cueTokenText(words []timedWord, exact bool) string {
	parts := make([]string, 0, len(words))
	for _, word := range words {
		parts = append(parts, word.Text)
	}
	separator := " "
	if exact {
		separator = ""
	}
	return strings.TrimSpace(strings.Join(parts, separator))
}

func hasSentenceEnding(value string) bool {
	value = strings.TrimSpace(value)
	for _, suffix := range []string{".", "?", "!", "。", "？", "！"} {
		if strings.HasSuffix(value, suffix) {
			return true
		}
	}
	return false
}
