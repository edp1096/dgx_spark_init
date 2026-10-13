package server

import (
	"archive/zip"
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"golang.org/x/net/html"
	"sparktalk/internal/config"
	"sparktalk/internal/llm"
	"sparktalk/internal/sourcebook"
)

// Exercise generation, durable checkpoints and all base exports in a temporary
// database; the production database is consulted read-only for admission.
func TestDocumentLayoutLive(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_LAYOUT_LIVE")
	if dir == "" {
		t.Skip("opt-in formatted document translation")
	}
	production, err := sql.Open("sqlite", "file:"+os.Getenv("SPARKTALK_DOCUMENT_LAYOUT_DB")+"?mode=ro")
	if err != nil {
		t.Fatal(err)
	}
	defer production.Close()
	idle := func() bool {
		var active int
		err := production.QueryRow(`SELECT (SELECT count(*) FROM workflow_runs WHERE status='running')+(SELECT count(*) FROM messages WHERE status NOT IN ('completed','cancelled','failed'))`).Scan(&active)
		return err == nil && active == 0
	}
	if !idle() {
		t.Fatal("user task is active")
	}
	response, err := http.Get("http://127.0.0.1:8585/api/config")
	if err != nil {
		t.Fatal(err)
	}
	var cfg config.Config
	err = json.NewDecoder(response.Body).Decode(&cfg)
	response.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Model.DefaultModel != "qwen38fn_exl3" {
		t.Fatal("active model changed")
	}
	s, _ := documentFixture(t, func(http.ResponseWriter, *http.Request) { t.Error("mock endpoint must not be used") })
	s.cfg.Model, s.cfg.Context = cfg.Model, cfg.Context
	s.llm = llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType)
	raw := []byte(`<html><body><h1>A quiet visit</h1><p>At 9:30, Mina visited <a href="#house"><span>the red </span><span>house</span></a>. She <strong>did not open the door</strong>. Read <a href="#note">the reference</a> carefully.<sup><a href="#note">1</a></sup></p><p id="house">The house stood beside a river. <em>Only seven blue birds</em> were on its roof.</p><p>The formula is <code>x = 7</code>.</p><p>A visitor waited here.<a id="page2"></a>The next page begins here.</p><figure><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==" alt="A red square"><figcaption>A small illustration.</figcaption></figure><table><tr><th>Place</th><th>Birds</th></tr><tr><td>Roof</td><td>7</td></tr></table><p class="poem">The river is quiet.<br>The blue birds sing.</p><p id="note">1. The visitor left the door closed.</p><p>彼女は<ruby><rb>川</rb><rp>(</rp><rt>かわ</rt><rp>)</rp></ruby>を見た。</p></body></html>`)
	b, err := sourcebook.Parse(raw, "text/html", "https://example.org/layout", nil)
	if err != nil {
		t.Fatal(err)
	}
	source, err := s.storeSource("s", b, raw)
	if err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{source}, Effort: "low", Glossary: map[string]string{"Mina": "미나"}, Instructions: "한국어로 빠짐없이 자연스럽게 번역하세요. 숫자와 부정 표현, 링크와 강조, 표와 시의 행 구분을 보존하세요."})
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Minute)
	for run.Status == "running" && time.Now().Before(deadline) {
		if !idle() {
			s.cancelSessionDocuments(context.Background(), "s")
			t.Fatal("user work started; isolated task canceled")
		}
		time.Sleep(time.Second)
		run, err = s.db.DocumentRun("s", run.ID)
		if err != nil {
			t.Fatal(err)
		}
	}
	if !run.DocumentFinished() {
		t.Fatal("document did not finish", run.Status, run.Document.Error)
	}
	units, err := s.db.DocumentUnits("s", run.ID)
	if err != nil {
		t.Fatal(err)
	}
	for _, unit := range units {
		if err := validateDocumentCoverage(unit, unit.Output); err != nil {
			t.Fatal(err)
		}
	}
	for _, format := range []string{"txt", "md", "html", "zip"} {
		data, _, err := s.db.DocumentOutput("s", run.ID, "output."+format)
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(dir, "output."+format), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	page, _ := os.ReadFile(filepath.Join(dir, "output.html"))
	doc, err := html.Parse(bytes.NewReader(page))
	if err != nil {
		t.Fatal(err)
	}
	var text func(*html.Node) string
	text = func(n *html.Node) string {
		if n.Type == html.TextNode {
			return n.Data
		}
		var s strings.Builder
		for child := n.FirstChild; child != nil; child = child.NextSibling {
			s.WriteString(text(child))
		}
		return s.String()
	}
	counts := map[string]int{}
	protected := map[string][]string{}
	ids := map[string]bool{}
	hrefs := []string{}
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			counts[n.Data]++
			for _, attr := range n.Attr {
				if attr.Key == "id" {
					ids[attr.Val] = true
				}
				if n.Data == "a" && attr.Key == "href" {
					hrefs = append(hrefs, attr.Val)
				}
			}
			switch n.Data {
			case "a", "strong", "em", "code", "sup":
				if strings.TrimSpace(text(n)) != "" {
					protected[n.Data] = append(protected[n.Data], text(n))
				} else if n.Data != "a" {
					t.Error("empty protected element", n.Data)
				}
			}
		}
		for child := n.FirstChild; child != nil; child = child.NextSibling {
			walk(child)
		}
	}
	walk(doc)
	if len(protected["a"]) != 3 || len(protected["strong"]) != 1 || len(protected["em"]) != 1 || len(protected["code"]) != 1 || counts["table"] != 1 || counts["br"] != 1 || counts["img"] != 1 {
		t.Fatal("lost HTML structure", counts, protected)
	}
	if fmt.Sprint(hrefs) != "[#house #note #note]" || !ids["house"] || !ids["note"] || !ids["page2"] {
		t.Fatal("anchor targets changed", hrefs, ids)
	}
	archive, _ := os.ReadFile(filepath.Join(dir, "output.zip"))
	z, err := zip.NewReader(bytes.NewReader(archive), int64(len(archive)))
	if err != nil {
		t.Fatal(err)
	}
	images := 0
	for _, file := range z.File {
		if strings.HasPrefix(file.Name, "images/") {
			images++
		}
	}
	if images != 1 || !bytes.Contains(page, []byte("data:image/png;base64,")) {
		t.Fatal("illustration missing")
	}
	report := map[string]any{"seconds": time.Since(started).Seconds(), "run": run, "units": units, "protected_text": protected, "html_counts": counts, "hrefs": hrefs, "anchor_ids": ids, "zip_images": images, "production_db_read_only": true}
	data, _ := json.MarshalIndent(report, "", "  ")
	if err = os.WriteFile(filepath.Join(dir, "report.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	t.Log(fmt.Sprintf("layout task completed: %d calls, %.2fs, protected text=%v", run.Document.Calls, time.Since(started).Seconds(), protected))
}
