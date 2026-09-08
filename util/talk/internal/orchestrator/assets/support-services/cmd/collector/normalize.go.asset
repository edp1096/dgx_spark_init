package main

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"path/filepath"
	"strings"
	"time"
	"unicode"

	"golang.org/x/net/html"
)

type manifest struct {
	Version      int       `json:"version"`
	RequestedURL string    `json:"requested_url"`
	FinalURL     string    `json:"final_url"`
	Title        string    `json:"title"`
	Method       string    `json:"method"`
	ContentType  string    `json:"content_type"`
	RawPath      string    `json:"raw_path"`
	FetchedAt    time.Time `json:"fetched_at"`
}

type linkRecord struct {
	Text string `json:"text"`
	URL  string `json:"url"`
}

type resourceRecord struct {
	URL      string `json:"url"`
	MIMEType string `json:"mime_type,omitempty"`
	Type     string `json:"type,omitempty"`
	Status   int64  `json:"status,omitempty"`
}

type collected struct {
	Manifest    manifest
	Raw         []byte
	Text        string
	Tables      [][][]string
	Links       []linkRecord
	Resources   []resourceRecord
	Publication *publicationPlan
	Screenshot  []byte
}

func normalizeHTML(data []byte, baseURL string) (string, string, [][][]string, []linkRecord, error) {
	document, err := html.Parse(bytes.NewReader(data))
	if err != nil {
		return "", "", nil, nil, err
	}
	base, _ := url.Parse(baseURL)
	title := ""
	var text strings.Builder
	links := []linkRecord{}
	tables := [][][]string{}
	var visit func(*html.Node, bool, bool)
	visit = func(node *html.Node, hidden, insideTable bool) {
		if node.Type == html.ElementNode {
			switch node.Data {
			case "script", "style", "noscript", "svg", "template":
				hidden = true
			case "table":
				if table := parseTable(node); len(table) > 0 {
					tables = append(tables, table)
				}
				insideTable = true
			case "a":
				if href := attribute(node, "href"); href != "" {
					if target, err := base.Parse(href); err == nil && (target.Scheme == "http" || target.Scheme == "https") {
						links = append(links, linkRecord{Text: compactText(nodeText(node)), URL: target.String()})
					}
				}
			}
		}
		if !hidden && !insideTable && node.Type == html.TextNode {
			value := compactText(node.Data)
			if value != "" {
				if node.Parent != nil && node.Parent.Type == html.ElementNode && node.Parent.Data == "title" {
					title = value
				}
				text.WriteString(value)
				text.WriteByte('\n')
			}
		}
		for child := node.FirstChild; child != nil; child = child.NextSibling {
			visit(child, hidden, insideTable)
		}
	}
	visit(document, false, false)
	for index, table := range tables {
		text.WriteString(fmt.Sprintf("\n[표 %d]\n", index+1))
		for _, row := range table {
			text.WriteString(strings.Join(row, " | "))
			text.WriteByte('\n')
		}
	}
	return title, strings.TrimSpace(text.String()), tables, dedupeLinks(links), nil
}

func parseTable(table *html.Node) [][]string {
	rows := [][]string{}
	var walk func(*html.Node)
	walk = func(node *html.Node) {
		if node.Type == html.ElementNode && node.Data == "tr" {
			row := []string{}
			for child := node.FirstChild; child != nil; child = child.NextSibling {
				if child.Type == html.ElementNode && (child.Data == "td" || child.Data == "th") {
					row = append(row, compactText(nodeText(child)))
				}
			}
			if len(row) > 0 {
				rows = append(rows, row)
			}
			return
		}
		for child := node.FirstChild; child != nil; child = child.NextSibling {
			walk(child)
		}
	}
	walk(table)
	return rows
}

func nodeText(node *html.Node) string {
	var output strings.Builder
	var walk func(*html.Node)
	walk = func(current *html.Node) {
		if current.Type == html.TextNode {
			output.WriteString(current.Data)
			output.WriteByte(' ')
		}
		for child := current.FirstChild; child != nil; child = child.NextSibling {
			walk(child)
		}
	}
	walk(node)
	return output.String()
}

func compactText(value string) string {
	return strings.Join(strings.Fields(value), " ")
}

func attribute(node *html.Node, name string) string {
	for _, attribute := range node.Attr {
		if attribute.Key == name {
			return strings.TrimSpace(attribute.Val)
		}
	}
	return ""
}

func dedupeLinks(items []linkRecord) []linkRecord {
	seen := make(map[string]struct{})
	result := make([]linkRecord, 0, len(items))
	for _, item := range items {
		if _, ok := seen[item.URL]; ok {
			continue
		}
		seen[item.URL] = struct{}{}
		result = append(result, item)
		if len(result) >= 2000 {
			break
		}
	}
	return result
}

func writeBundle(output io.Writer, item collected) error {
	archive := zip.NewWriter(output)
	write := func(name string, data []byte) error {
		header := &zip.FileHeader{Name: name, Method: zip.Deflate}
		header.SetModTime(item.Manifest.FetchedAt)
		part, err := archive.CreateHeader(header)
		if err != nil {
			return err
		}
		_, err = part.Write(data)
		return err
	}
	manifestData, _ := json.MarshalIndent(item.Manifest, "", "  ")
	if err := write("manifest.json", manifestData); err != nil {
		return err
	}
	if err := write(item.Manifest.RawPath, item.Raw); err != nil {
		return err
	}
	if item.Text != "" {
		if err := write("normalized/text.txt", []byte(item.Text)); err != nil {
			return err
		}
	}
	if len(item.Tables) > 0 {
		data, _ := json.MarshalIndent(item.Tables, "", "  ")
		if err := write("normalized/tables.json", data); err != nil {
			return err
		}
	}
	if len(item.Links) > 0 {
		data, _ := json.MarshalIndent(item.Links, "", "  ")
		if err := write("normalized/links.json", data); err != nil {
			return err
		}
	}
	if len(item.Resources) > 0 {
		data, _ := json.MarshalIndent(item.Resources, "", "  ")
		if err := write("normalized/resources.json", data); err != nil {
			return err
		}
	}
	if item.Publication != nil {
		data, _ := json.MarshalIndent(item.Publication, "", "  ")
		if err := write("normalized/publication.json", data); err != nil {
			return err
		}
	}
	if len(item.Screenshot) > 0 {
		if err := write("preview/screenshot.png", item.Screenshot); err != nil {
			return err
		}
	}
	return archive.Close()
}

func rawPath(rawURL, contentType string) string {
	extension := extensionForContentType(contentType)
	if parsed, err := url.Parse(rawURL); err == nil {
		if ext := filepath.Ext(parsed.Path); ext != "" && len(ext) <= 10 {
			extension = ext
		}
	}
	if extension == "" {
		extension = ".bin"
	}
	return "raw/source" + extension
}

func extensionForContentType(value string) string {
	value = strings.ToLower(strings.Split(value, ";")[0])
	switch value {
	case "text/html":
		return ".html"
	case "application/pdf":
		return ".pdf"
	case "application/json":
		return ".json"
	case "application/javascript", "application/x-javascript", "text/javascript":
		return ".js"
	case "text/csv":
		return ".csv"
	case "text/plain":
		return ".txt"
	case "application/xml", "text/xml":
		return ".xml"
	case "image/png":
		return ".png"
	case "image/jpeg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	default:
		return ""
	}
}

func fallbackTitle(rawURL string) string {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return "수집 자료"
	}
	name := strings.TrimSpace(filepath.Base(parsed.Path))
	if name == "" || name == "." || name == "/" {
		name = parsed.Hostname()
	}
	if decoded, err := url.PathUnescape(name); err == nil {
		name = decoded
	}
	if name == "" {
		return "수집 자료"
	}
	return name
}

func bundleName(title string) string {
	title = strings.Map(func(char rune) rune {
		if char == '/' || char == '\\' || unicode.IsControl(char) {
			return -1
		}
		return char
	}, title)
	title = strings.TrimSpace(title)
	if title == "" {
		title = "collection"
	}
	return fmt.Sprintf("%s.zip", title)
}
