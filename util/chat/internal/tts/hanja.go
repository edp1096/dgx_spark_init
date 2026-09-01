package tts

import (
	"bufio"
	"bytes"
	"compress/gzip"
	_ "embed"
	"strconv"
	"strings"
	"sync"
	"unicode"
)

// Generated from Unicode 17.0.0 Unihan_Readings.txt kHangul values.
// See assets/LICENSE.UNICODE and cmd/gen_hanja.
//
//go:embed assets/hanja_readings.tsv.gz
var hanjaReadingsGzip []byte

var (
	hanjaReadingsOnce sync.Once
	hanjaReadings     map[rune]string
)

func koreanizeHanja(text string) string {
	readings := loadHanjaReadings()
	var result strings.Builder
	result.Grow(len(text))
	spanStart := true
	for _, character := range text {
		if !unicode.Is(unicode.Han, character) {
			result.WriteRune(character)
			spanStart = true
			continue
		}
		reading, ok := readings[character]
		if !ok {
			result.WriteRune(character)
			spanStart = false
			continue
		}
		if spanStart {
			reading = applyInitialSoundLaw(reading)
		}
		result.WriteString(reading)
		spanStart = false
	}
	return result.String()
}

func loadHanjaReadings() map[rune]string {
	hanjaReadingsOnce.Do(func() {
		hanjaReadings = make(map[rune]string, 8500)
		reader, err := gzip.NewReader(bytes.NewReader(hanjaReadingsGzip))
		if err != nil {
			panic("open embedded Unihan readings: " + err.Error())
		}
		defer reader.Close()
		scanner := bufio.NewScanner(reader)
		for scanner.Scan() {
			fields := strings.SplitN(scanner.Text(), "\t", 2)
			if len(fields) != 2 {
				continue
			}
			codepoint, parseErr := strconv.ParseInt(fields[0], 16, 32)
			if parseErr == nil {
				hanjaReadings[rune(codepoint)] = fields[1]
			}
		}
		if err := scanner.Err(); err != nil {
			panic("read embedded Unihan readings: " + err.Error())
		}
	})
	return hanjaReadings
}

func applyInitialSoundLaw(reading string) string {
	runes := []rune(reading)
	if len(runes) == 0 || runes[0] < 0xAC00 || runes[0] > 0xD7A3 {
		return reading
	}
	const syllableBase = 0xAC00
	index := int(runes[0] - syllableBase)
	initial := index / (21 * 28)
	vowel := (index / 28) % 21
	final := index % 28
	newInitial := initial
	// ㄴ + ㅕ/ㅛ/ㅠ/ㅣ -> ㅇ
	if initial == 2 && (vowel == 6 || vowel == 12 || vowel == 17 || vowel == 20) {
		newInitial = 11
	}
	if initial == 5 { // ㄹ
		if vowel == 2 || vowel == 6 || vowel == 7 || vowel == 12 || vowel == 17 || vowel == 20 {
			newInitial = 11 // ㅑ/ㅕ/ㅖ/ㅛ/ㅠ/ㅣ -> ㅇ
		} else {
			newInitial = 2 // remaining vowels -> ㄴ
		}
	}
	if newInitial != initial {
		runes[0] = rune(syllableBase + newInitial*21*28 + vowel*28 + final)
	}
	return string(runes)
}
