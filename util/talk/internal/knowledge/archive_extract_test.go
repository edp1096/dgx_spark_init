package knowledge

import (
	"archive/zip"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExtractDOCXAndHWPXText(t *testing.T) {
	for _, test := range []struct {
		name, mime string
		entries    map[string]string
		want       string
	}{
		{"manual.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", map[string]string{
			"word/document.xml": `<w:document xmlns:w="w"><w:body><w:p><w:r><w:t>첫 문장</w:t></w:r></w:p><w:p><w:r><w:t>둘째 문장</w:t></w:r></w:p></w:body></w:document>`,
		}, "첫 문장\n둘째 문장"},
		{"book.hwpx", "application/vnd.hancom.hwpx", map[string]string{
			"Contents/content.hpf": `<package/>`, "Contents/section0.xml": `<section><p><t>한글 문서 본문</t></p></section>`,
		}, "한글 문서 본문"},
	} {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), test.name)
			writeKnowledgeArchive(t, path, test.entries)
			pages, err := (&Extractor{}).Extract(path, test.mime)
			if err != nil || len(pages) != 1 || !strings.Contains(pages[0].Text, test.want) {
				t.Fatalf("pages=%+v err=%v", pages, err)
			}
		})
	}
}

func TestExtractXLSXSharedStrings(t *testing.T) {
	path := filepath.Join(t.TempDir(), "table.xlsx")
	writeKnowledgeArchive(t, path, map[string]string{
		"xl/workbook.xml":          `<workbook/>`,
		"xl/sharedStrings.xml":     `<sst><si><t>부품</t></si><si><t>용량</t></si></sst>`,
		"xl/worksheets/sheet1.xml": `<worksheet><sheetData><row><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row><row><c r="A2" t="inlineStr"><is><t>MLCC</t></is></c><c r="B2"><v>10</v></c></row></sheetData></worksheet>`,
	})
	pages, err := (&Extractor{}).Extract(path, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
	if err != nil || len(pages) != 1 || !strings.Contains(pages[0].Text, "A1=부품 | B1=용량") || !strings.Contains(pages[0].Text, "A2=MLCC | B2=10") {
		t.Fatalf("pages=%+v err=%v", pages, err)
	}
}

func writeKnowledgeArchive(t *testing.T, path string, entries map[string]string) {
	t.Helper()
	output, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	writer := zip.NewWriter(output)
	for name, value := range entries {
		file, err := writer.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := file.Write([]byte(value)); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	if err := output.Close(); err != nil {
		t.Fatal(err)
	}
}
