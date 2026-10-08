package knowledge

import (
	"strings"
	"unicode/utf8"

	"sparktalk/internal/db"
)

const (
	chunkRunes   = 1800
	overlapRunes = 180
)

func ChunkPages(documentID string, pages []Page) []db.KnowledgeChunk {
	chunks := []db.KnowledgeChunk{}
	for _, page := range pages {
		pageChunks := chunkPage(page.Text)
		for _, content := range pageChunks {
			chunks = append(chunks, db.KnowledgeChunk{
				DocumentID: documentID,
				Ordinal:    len(chunks),
				PageStart:  page.Number,
				PageEnd:    page.Number,
				Content:    content,
			})
		}
	}
	return chunks
}

func chunkPage(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	paragraphs := strings.Split(text, "\n\n")
	result := []string{}
	current := ""
	flush := func() {
		current = strings.TrimSpace(current)
		if current != "" {
			result = append(result, current)
		}
	}
	for _, paragraph := range paragraphs {
		paragraph = strings.TrimSpace(paragraph)
		if paragraph == "" {
			continue
		}
		if utf8.RuneCountInString(paragraph) > chunkRunes {
			if current != "" {
				flush()
				current = ""
			}
			result = append(result, splitLongText(paragraph)...)
			continue
		}
		candidate := paragraph
		if current != "" {
			candidate = current + "\n\n" + paragraph
		}
		if utf8.RuneCountInString(candidate) <= chunkRunes {
			current = candidate
			continue
		}
		previous := current
		flush()
		current = strings.TrimSpace(runeTail(previous, overlapRunes) + "\n\n" + paragraph)
	}
	if current != "" {
		flush()
	}
	return result
}

func splitLongText(text string) []string {
	runes := []rune(text)
	result := []string{}
	for start := 0; start < len(runes); {
		end := min(start+chunkRunes, len(runes))
		result = append(result, strings.TrimSpace(string(runes[start:end])))
		if end == len(runes) {
			break
		}
		start = max(end-overlapRunes, start+1)
	}
	return result
}

func runeTail(value string, limit int) string {
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	return string(runes[len(runes)-limit:])
}
