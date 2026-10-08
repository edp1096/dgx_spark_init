package knowledge

import (
	"archive/zip"
	"encoding/xml"
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const maxArchiveEntryBytes int64 = 32 << 20

func extractArchiveDocument(path, mimeType string) ([]Page, error) {
	archive, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("open document archive: %w", err)
	}
	defer archive.Close()
	if len(archive.File) > 10000 {
		return nil, fmt.Errorf("document archive has too many entries")
	}
	switch mimeType {
	case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
		return extractXMLArchivePages(archive.File, func(name string) bool {
			base := filepath.Base(name)
			return name == "word/document.xml" || strings.HasPrefix(name, "word/header") && strings.HasSuffix(name, ".xml") ||
				strings.HasPrefix(name, "word/footer") && strings.HasSuffix(name, ".xml") || base == "footnotes.xml" || base == "endnotes.xml"
		})
	case "application/vnd.openxmlformats-officedocument.presentationml.presentation":
		return extractXMLArchivePages(archive.File, func(name string) bool {
			return strings.HasPrefix(name, "ppt/slides/slide") && strings.HasSuffix(name, ".xml")
		})
	case "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
		return extractSpreadsheetPages(archive.File)
	case "application/vnd.hancom.hwpx":
		return extractXMLArchivePages(archive.File, func(name string) bool {
			return strings.HasPrefix(name, "Contents/section") && strings.HasSuffix(name, ".xml")
		})
	case "application/vnd.oasis.opendocument.text", "application/vnd.oasis.opendocument.presentation", "application/vnd.oasis.opendocument.spreadsheet":
		return extractXMLArchivePages(archive.File, func(name string) bool { return name == "content.xml" })
	case "application/epub+zip":
		return extractXMLArchivePages(archive.File, func(name string) bool {
			extension := strings.ToLower(filepath.Ext(name))
			return extension == ".xhtml" || extension == ".html" || extension == ".htm"
		})
	default:
		return nil, fmt.Errorf("unsupported archive document type %s", mimeType)
	}
}

func extractXMLArchivePages(files []*zip.File, selectEntry func(string) bool) ([]Page, error) {
	selected := selectedArchiveEntries(files, selectEntry)
	if err := validateSelectedArchiveSize(selected); err != nil {
		return nil, err
	}
	pages := make([]Page, 0, len(selected))
	for _, file := range selected {
		data, err := readDocumentArchiveEntry(file)
		if err != nil {
			return nil, err
		}
		text, err := extractXMLText(data)
		if err != nil {
			return nil, fmt.Errorf("extract %s: %w", file.Name, err)
		}
		if text != "" {
			pages = append(pages, Page{Number: len(pages) + 1, Text: text})
		}
	}
	if len(pages) == 0 {
		return nil, ErrOCRRequired
	}
	return pages, nil
}

func selectedArchiveEntries(files []*zip.File, selectEntry func(string) bool) []*zip.File {
	selected := make([]*zip.File, 0)
	for _, file := range files {
		name := strings.TrimPrefix(strings.ReplaceAll(file.Name, "\\", "/"), "/")
		if !file.FileInfo().IsDir() && selectEntry(name) {
			selected = append(selected, file)
		}
	}
	sort.SliceStable(selected, func(i, j int) bool {
		left, right := archiveEntryNumber(selected[i].Name), archiveEntryNumber(selected[j].Name)
		if left != right && left >= 0 && right >= 0 {
			return left < right
		}
		return selected[i].Name < selected[j].Name
	})
	return selected
}

func archiveEntryNumber(name string) int {
	base := strings.TrimSuffix(filepath.Base(name), filepath.Ext(name))
	index := len(base)
	for index > 0 && base[index-1] >= '0' && base[index-1] <= '9' {
		index--
	}
	if index == len(base) {
		return -1
	}
	value, _ := strconv.Atoi(base[index:])
	return value
}

func readDocumentArchiveEntry(file *zip.File) ([]byte, error) {
	if file.UncompressedSize64 > uint64(maxArchiveEntryBytes) {
		return nil, fmt.Errorf("document entry %s exceeds %d MB", file.Name, maxArchiveEntryBytes>>20)
	}
	reader, err := file.Open()
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	data, err := io.ReadAll(io.LimitReader(reader, maxArchiveEntryBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxArchiveEntryBytes {
		return nil, fmt.Errorf("document entry %s is too large", file.Name)
	}
	return data, nil
}

func extractXMLText(data []byte) (string, error) {
	decoder := xml.NewDecoder(strings.NewReader(string(data)))
	var output strings.Builder
	lineStart := true
	for {
		token, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		switch value := token.(type) {
		case xml.CharData:
			text := strings.TrimSpace(string(value))
			if text != "" {
				if output.Len() > 0 && !lineStart {
					output.WriteByte(' ')
				}
				output.WriteString(text)
				lineStart = false
			}
		case xml.EndElement:
			switch strings.ToLower(value.Name.Local) {
			case "p", "h", "tr", "table-row", "section", "sld", "br":
				output.WriteByte('\n')
				lineStart = true
			}
		}
	}
	return normalizeText(output.String()), nil
}

func validateSelectedArchiveSize(files []*zip.File) error {
	var total uint64
	for _, file := range files {
		total += file.UncompressedSize64
		if file.UncompressedSize64 > uint64(maxArchiveEntryBytes) || total > uint64(MaxSourceBytes) {
			return fmt.Errorf("document archive expands beyond supported limits")
		}
	}
	return nil
}

func extractSpreadsheetPages(files []*zip.File) ([]Page, error) {
	shared := []string{}
	for _, file := range files {
		if file.Name != "xl/sharedStrings.xml" {
			continue
		}
		data, err := readDocumentArchiveEntry(file)
		if err != nil {
			return nil, err
		}
		shared, err = spreadsheetSharedStrings(data)
		if err != nil {
			return nil, err
		}
	}
	sheets := selectedArchiveEntries(files, func(name string) bool {
		return strings.HasPrefix(name, "xl/worksheets/sheet") && strings.HasSuffix(name, ".xml")
	})
	if err := validateSelectedArchiveSize(sheets); err != nil {
		return nil, err
	}
	pages := make([]Page, 0, len(sheets))
	for _, file := range sheets {
		data, err := readDocumentArchiveEntry(file)
		if err != nil {
			return nil, err
		}
		text, err := spreadsheetSheetText(data, shared)
		if err != nil {
			return nil, fmt.Errorf("extract %s: %w", file.Name, err)
		}
		if text != "" {
			pages = append(pages, Page{Number: len(pages) + 1, Text: text})
		}
	}
	if len(pages) == 0 {
		return nil, ErrOCRRequired
	}
	return pages, nil
}

func spreadsheetSharedStrings(data []byte) ([]string, error) {
	decoder := xml.NewDecoder(strings.NewReader(string(data)))
	values := []string{}
	var current strings.Builder
	inItem := false
	for {
		token, err := decoder.Token()
		if err == io.EOF {
			return values, nil
		}
		if err != nil {
			return nil, err
		}
		switch value := token.(type) {
		case xml.StartElement:
			if value.Name.Local == "si" {
				inItem = true
				current.Reset()
			}
		case xml.CharData:
			if inItem {
				current.Write(value)
			}
		case xml.EndElement:
			if value.Name.Local == "si" {
				values = append(values, strings.TrimSpace(current.String()))
				inItem = false
			}
		}
	}
}

func spreadsheetSheetText(data []byte, shared []string) (string, error) {
	decoder := xml.NewDecoder(strings.NewReader(string(data)))
	var output strings.Builder
	cellRef, cellType, cellValue := "", "", ""
	inValue := false
	for {
		token, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		switch value := token.(type) {
		case xml.StartElement:
			if value.Name.Local == "c" {
				cellRef, cellType, cellValue = "", "", ""
				for _, attribute := range value.Attr {
					if attribute.Name.Local == "r" {
						cellRef = attribute.Value
					}
					if attribute.Name.Local == "t" {
						cellType = attribute.Value
					}
				}
			}
			if value.Name.Local == "v" || value.Name.Local == "t" {
				inValue = true
			}
		case xml.CharData:
			if inValue {
				cellValue += string(value)
			}
		case xml.EndElement:
			if value.Name.Local == "v" || value.Name.Local == "t" {
				inValue = false
			}
			if value.Name.Local == "c" {
				cellValue = strings.TrimSpace(cellValue)
				if cellType == "s" {
					index, _ := strconv.Atoi(cellValue)
					if index >= 0 && index < len(shared) {
						cellValue = shared[index]
					}
				}
				if cellValue != "" {
					if output.Len() > 0 {
						output.WriteString(" | ")
					}
					if cellRef != "" {
						output.WriteString(cellRef + "=")
					}
					output.WriteString(cellValue)
				}
			}
			if value.Name.Local == "row" {
				output.WriteByte('\n')
			}
		}
	}
	return normalizeText(output.String()), nil
}
