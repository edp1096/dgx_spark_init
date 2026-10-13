package server

import (
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

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
	"sparktalk/internal/orchestrator"
	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

// Opt-in integration check. Uses a separate DB; never changes the running Talk
// configuration, model lifecycle, or user conversations.
func TestDocumentTaskLive(t *testing.T) {
	mode := os.Getenv("SPARKTALK_DOCUMENT_LIVE")
	if mode == "" {
		t.Skip("opt-in live model validation")
	}
	dir := os.Getenv("SPARKTALK_DOCUMENT_REPORT")
	if dir == "" {
		dir = t.TempDir()
	}
	if e := os.MkdirAll(dir, 0700); e != nil {
		t.Fatal(e)
	}
	resp, e := http.Get("http://127.0.0.1:8585/api/config")
	if e != nil {
		t.Fatal(e)
	}
	defer resp.Body.Close()
	var cfg config.Config
	if e = json.NewDecoder(resp.Body).Decode(&cfg); e != nil {
		t.Fatal(e)
	}
	database, e := db.Open(filepath.Join(dir, "validation.db"))
	if e != nil {
		t.Fatal(e)
	}
	defer database.Close()
	s := &Server{db: database, cfg: cfg, llm: llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType), knowledgeIndex: nil}
	defer func() {
		s.tasks.Stop()
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		s.tasks.Wait(ctx)
	}()
	cases := []struct{ name, url, kind string }{{"smoke", "", "translate"}}
	if mode == "full" {
		cases = []struct{ name, url, kind string }{{"twilight", "https://www.gutenberg.org/cache/epub/79755/pg79755.txt", "translate"}, {"alice", "https://www.gutenberg.org/cache/epub/11/pg11-images.html", "translate"}}
	}
	type resumeItem struct {
		Session string `json:"session_id"`
		Run     string `json:"run_id"`
		Source  string `json:"source_id"`
	}
	resume := map[string]resumeItem{}
	if data, err := os.ReadFile(filepath.Join(dir, "resume.json")); err == nil {
		if err = json.Unmarshal(data, &resume); err != nil {
			t.Fatal(err)
		}
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			prior, hasPrior := resume[tc.name]
			session := prior.Session
			if !hasPrior {
				session = "live-" + tc.name + "-" + newID()
				if _, e = database.CreateSession(session, tc.name, cfg.Model.DefaultModel, "off"); e != nil {
					t.Fatal(e)
				}
			}
			var id string
			var b *sourcebook.Book
			if hasPrior {
				id = prior.Source
				b, e = database.Source(session, id, false)
			} else if tc.url != "" {
				id, b, e = s.openURLSource(context.Background(), session, tc.url)
			} else {
				raw := []byte("CHAPTER I.\n\nAlice sat beside her sister. She saw a white rabbit carrying a watch.\n\nCHAPTER II.\n\nThe rabbit disappeared into a hole. Alice followed him and began her adventure.")
				b, e = sourcebook.Parse(raw, "text/plain", "", nil)
				if e == nil {
					id, e = s.storeSource(session, b, raw)
				}
			}
			if e != nil {
				t.Fatal(e)
			}
			t.Logf("source=%s chars=%d blocks=%d images=%d", id, len([]rune(b.Text)), len(b.Blocks), len(b.Assets))
			var run workflows.Run
			initialCompleted := 0
			if hasPrior {
				run, e = database.DocumentRun(session, prior.Run)
				if e == nil {
					initialCompleted = run.Document.Completed
					if run.Status != "completed" {
						run, e = s.documentAction(context.Background(), session, documentStart{Action: "resume", RunID: prior.Run})
					}
				}
			} else {
				run, e = s.documentAction(context.Background(), session, documentStart{Action: "start", Effort: "none", Kind: tc.kind, Sources: []string{id}, Instructions: "원문 전체를 생략하거나 요약하지 말고 한국어 문어체로 충실히 번역한다. 설명이나 진행 제안을 번역문에 넣지 않는다."})
			}
			t.Logf("resuming from %d completed units", initialCompleted)
			if e != nil {
				t.Fatal(e)
			}
			ticker := time.NewTicker(10 * time.Second)
			defer ticker.Stop()
			deadline := time.Now().Add(3 * time.Hour)
			for {
				run, e = database.DocumentRun(session, run.ID)
				if e != nil {
					t.Fatal(e)
				}
				if run.Status != "running" {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("live validation deadline")
				}
				<-ticker.C
				t.Logf("%s %d/%d calls=%d", tc.name, run.Document.Completed, run.Document.Total, run.Document.Calls)
			}
			units, _ := database.DocumentUnits(session, run.ID)
			report := map[string]any{"initial_completed": initialCompleted, "run": run, "units": units, "source_chars": len([]rune(b.Text)), "source_blocks": len(b.Blocks), "source_images": len(b.Assets)}
			data, _ := json.MarshalIndent(report, "", "  ")
			if e = os.WriteFile(filepath.Join(dir, tc.name+".json"), data, 0600); e != nil {
				t.Fatal(e)
			}
			for _, ext := range []string{"txt", "md", "html", "zip"} {
				data, _, err := database.DocumentOutput(session, run.ID, "output."+ext)
				if err == nil {
					os.WriteFile(filepath.Join(dir, tc.name+"."+ext), data, 0600)
				}
			}
			if run.Status != "completed" && !(run.Status == "needs_review" && run.Current == 2) {
				t.Fatalf("live task stopped: %s %s", run.Status, run.Document.Error)
			}
			fmt.Printf("LIVE_DOCUMENT_RESULT %s %s %d/%d\n", tc.name, run.Status, run.Document.Completed, run.Document.Total)
		})
	}
}

func TestDocumentDispatchLive(t *testing.T) {
	if os.Getenv("SPARKTALK_DOCUMENT_DISPATCH") == "" {
		t.Skip("opt-in dispatch validation")
	}
	resp, e := http.Get("http://127.0.0.1:8585/api/config")
	if e != nil {
		t.Fatal(e)
	}
	defer resp.Body.Close()
	var cfg config.Config
	if e = json.NewDecoder(resp.Body).Decode(&cfg); e != nil {
		t.Fatal(e)
	}
	database, e := db.Open(filepath.Join(t.TempDir(), "dispatch.db"))
	if e != nil {
		t.Fatal(e)
	}
	defer database.Close()
	database.CreateSession("s", "dispatch", cfg.Model.DefaultModel, "none")
	s := &Server{db: database, cfg: cfg, llm: llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType)}
	s.cfg.Extra = config.ExtraConfig{}
	s.cfg.Memory.Enabled = false
	s.cfg.Tools = config.ToolsConfig{Enabled: true, MaxRounds: 6, Timeout: "15s", SearchResults: 3}
	defer func() {
		s.tasks.Stop()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		s.tasks.Wait(ctx)
	}()
	body := strings.Repeat("A rabbit carried a watch into the garden. Alice followed the rabbit and found a door.\n\n", 160)
	b, e := sourcebook.Parse([]byte(body), "text/plain", "", nil)
	if e != nil {
		t.Fatal(e)
	}
	id, e := s.storeSource("s", b, []byte(body))
	if e != nil {
		t.Fatal(e)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	ctx, release, e := s.trackGeneration(ctx)
	if e != nil {
		t.Fatal(e)
	}
	defer release()
	messages := []llm.Message{{Role: "user", Content: fmt.Sprintf("이미 보관된 문서 원문 전체를 한국어로 번역하고 HTML 파일로 만들어라. source_id=%s, total_chars=%d. 작업을 나눠서 끝까지 처리해라. 원문 미리보기: %s", id, len([]rune(b.Text)), body[:200])}}
	result, e := runCompletionLoopForSession(s, "s", llm.WithOutputLimit(ctx, 16384), s.llm, messages, cfg.Model.DefaultModel, "none", "", s.cfg.Tools, true, func(string, any) error { return nil })
	if e != nil {
		t.Fatal(e)
	}
	runs, e := database.WorkflowRuns("s")
	if e != nil || len(runs) != 1 || runs[0].Document == nil || runs[0].Document.Kind != "translate" || runs[0].Document.Start != 0 || runs[0].Document.End != 0 {
		t.Fatalf("model did not dispatch whole document: %s %v", result.Content, e)
	}
	if _, e = s.documentAction(context.Background(), "s", documentStart{Action: "pause", RunID: runs[0].ID}); e != nil {
		t.Fatal(e)
	}
	report, _ := json.MarshalIndent(map[string]any{"content": result.Content, "tool_trace": result.ToolTrace, "performance": result.Performance, "run": runs[0]}, "", "  ")
	dir := os.Getenv("SPARKTALK_DOCUMENT_REPORT")
	if dir != "" {
		os.MkdirAll(dir, 0700)
		os.WriteFile(filepath.Join(dir, "dispatch.json"), report, 0600)
	}
	t.Log(result.Content)
}

func TestDocumentQualitySampleLive(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_QUALITY")
	if dir == "" {
		t.Skip("opt-in quality sampling")
	}
	raw, e := os.ReadFile(filepath.Join(dir, "resume.json"))
	if e != nil {
		t.Fatal(e)
	}
	var resume map[string]struct {
		Run string `json:"run_id"`
	}
	if e = json.Unmarshal(raw, &resume); e != nil {
		t.Fatal(e)
	}
	// Read-only connection: no startup recovery/migration against the active trial.
	database, e := sql.Open("sqlite", "file:"+filepath.Join(dir, "validation.db")+"?mode=ro&_pragma=busy_timeout(5000)")
	if e != nil {
		t.Fatal(e)
	}
	defer database.Close()
	var run workflows.Run
	if e = database.QueryRow(`SELECT data FROM workflow_runs WHERE id=?`, resume["twilight"].Run).Scan(&raw); e != nil {
		t.Fatal(e)
	}
	if e = json.Unmarshal(raw, &run); e != nil {
		t.Fatal(e)
	}
	var u workflows.DocumentUnit
	if e = database.QueryRow(`SELECT data FROM workflow_units WHERE run_id=? AND ordinal=80`, run.ID).Scan(&raw); e != nil {
		t.Fatal(e)
	}
	json.Unmarshal(raw, &u)
	u.Inputs = u.Inputs[:min(2, len(u.Inputs))]
	u.OutputBudget = 0
	run.Document.Glossary = nil
	run.Document.GlossaryKinds = nil
	run.Document.ContextTail = ""
	resp, e := http.Get("http://127.0.0.1:8585/api/config")
	if e != nil {
		t.Fatal(e)
	}
	defer resp.Body.Close()
	var cfg config.Config
	json.NewDecoder(resp.Body).Decode(&cfg)
	client := llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType)
	result := map[string]any{"inputs": u.Inputs}
	for _, effort := range []string{"none", cfg.Model.ReasoningEffort} {
		run.Document.Effort = effort
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		start := time.Now()
		reply, err := requestDocumentUnit(ctx, client, run, u, cfg.Context.WindowTokens)
		cancel()
		if err != nil {
			t.Fatal(err)
		}
		output, err := validateDocumentOutput(u, "translate", reply)
		if err != nil {
			t.Fatal(err)
		}
		result[effort] = map[string]any{"output": output, "seconds": time.Since(start).Seconds(), "usage": reply.Usage, "reasoning_chars": len([]rune(reply.Reasoning)), "finish_reason": reply.FinishReason}
		t.Log(effort, time.Since(start))
	}
	data, _ := json.MarshalIndent(result, "", "  ")
	if e = os.WriteFile(filepath.Join(dir, "quality-sample.json"), data, 0600); e != nil {
		t.Fatal(e)
	}
}

// Final-code recovery and export check over the completed long-running trial.
func TestDocumentRepairLive(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_REPAIR")
	if dir == "" {
		t.Skip("opt-in repair of isolated validation DB")
	}
	path := filepath.Join(dir, "validation.db")
	ro, err := sql.Open("sqlite", "file:"+path+"?mode=ro")
	if err != nil {
		t.Fatal(err)
	}
	var active int
	err = ro.QueryRow(`SELECT count(*) FROM workflow_runs WHERE status='running'`).Scan(&active)
	ro.Close()
	if err != nil || active != 0 {
		t.Fatal("trial must be idle before reopening its DB", err, active)
	}
	resp, err := http.Get("http://127.0.0.1:8585/api/config")
	if err != nil {
		t.Fatal(err)
	}
	var cfg config.Config
	err = json.NewDecoder(resp.Body).Decode(&cfg)
	resp.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	database, err := db.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	s := &Server{db: database, cfg: cfg, llm: llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType)}
	defer func() {
		s.tasks.Stop()
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		s.tasks.Wait(ctx)
	}()
	var resume map[string]struct {
		Session string `json:"session_id"`
		Run     string `json:"run_id"`
		Source  string `json:"source_id"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "resume.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &resume); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"twilight", "alice"} {
		t.Run(name, func(t *testing.T) {
			item := resume[name]
			run, err := database.DocumentRun(item.Session, item.Run)
			if err != nil {
				t.Fatal(err)
			}
			before, _ := database.DocumentUnits(item.Session, item.Run)
			good := map[string]string{}
			bad := map[string]bool{}
			for _, u := range before {
				for id := range shortDocumentOutputs(u) {
					bad[id] = true
				}
				for id, text := range u.Output {
					if !bad[id] {
						good[id] = text
					}
				}
			}
			callsBefore := run.Document.Calls
			if len(bad) > 0 {
				run, err = s.documentAction(context.Background(), item.Session, documentStart{Action: "retry_quality", RunID: item.Run})
				if err != nil {
					t.Fatal(err)
				}
				deadline := time.Now().Add(20 * time.Minute)
				for run.Status == "running" && time.Now().Before(deadline) {
					time.Sleep(time.Second)
					run, err = database.DocumentRun(item.Session, item.Run)
					if err != nil {
						t.Fatal(err)
					}
				}
			}
			if run.Current != 2 || (run.Status != "completed" && run.Status != "needs_review") {
				t.Fatalf("repair failed: %s %s", run.Status, run.Document.Error)
			}
			units, _ := database.DocumentUnits(item.Session, item.Run)
			remaining := map[string]string{}
			for _, u := range units {
				for id, w := range shortDocumentOutputs(u) {
					remaining[id] = w
				}
				for id, text := range u.Output {
					if old, ok := good[id]; ok && old != text {
						t.Fatalf("good fragment changed: %s", id)
					}
					delete(good, id)
				}
			}
			if len(good) > 0 || len(remaining) > 0 {
				t.Fatalf("coverage/quality repair failed: retained=%d short=%d", len(good), len(remaining))
			}
			if err = s.exportDocument(context.Background(), run, units); err != nil {
				t.Fatal(err)
			}
			book, err := database.Source(item.Session, item.Source, true)
			if err != nil {
				t.Fatal(err)
			}
			report := map[string]any{"run": run, "units": units, "source_chars": len([]rune(book.Text)), "source_blocks": len(book.Blocks), "source_images": len(book.Assets), "repair_candidates": len(bad), "repair_calls": run.Document.Calls - callsBefore, "unchanged_good_fragments": true, "final_code_coverage_verified": true}
			data, _ := json.MarshalIndent(report, "", "  ")
			if err = os.WriteFile(filepath.Join(dir, name+"-final.json"), data, 0600); err != nil {
				t.Fatal(err)
			}
			for _, ext := range []string{"txt", "md", "html", "zip"} {
				data, _, err := database.DocumentOutput(item.Session, item.Run, "output."+ext)
				if err != nil {
					t.Fatal(err)
				}
				if err = os.WriteFile(filepath.Join(dir, name+"."+ext), data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			t.Logf("%s status=%s repaired=%d calls=%d/%d blocks=%d images=%d", name, run.Status, len(bad), callsBefore, run.Document.Calls, len(book.Blocks), len(book.Assets))
		})
	}
}

func TestDocumentConversionLive(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_CONVERT_LIVE")
	if dir == "" {
		t.Skip("opt-in conversion against installed Extra Documents")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	resp, err := http.Get("http://127.0.0.1:8585/api/config")
	if err != nil {
		t.Fatal(err)
	}
	var cfg config.Config
	err = json.NewDecoder(resp.Body).Decode(&cfg)
	resp.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	controller, err := orchestrator.NewControllerWithCatalog(*cfg.Runtime.Catalog)
	if err != nil {
		t.Fatal(err)
	}
	controller.ConfigurePaths(cfg.Runtime.DataDir, cfg.Runtime.ModelCache)
	defer controller.Close()
	for _, bundle := range controller.Catalog().Bundles {
		if bundle.ModelID == cfg.Model.DefaultModel {
			cfg.Runtime.ActiveBundle = bundle.ID
		}
	}
	defer func() {
		if err := controller.ComponentActionWithReserve("extra-documents", "stop", cfg.Runtime.MemoryReserveGiB, cfg.Runtime.ActiveBundle); err != nil {
			t.Error(err)
		}
	}()
	database, err := db.Open(filepath.Join(dir, "conversion.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	s := &Server{db: database, cfg: cfg, runtime: controller}
	session := "live-conversion-" + newID()
	if _, err = database.CreateSession(session, "live export", cfg.Model.DefaultModel, "off"); err != nil {
		t.Fatal(err)
	}
	run := workflows.Run{ID: newID(), SessionID: session, Status: "completed", Current: 2, Document: &workflows.DocumentTask{Kind: "translate", Model: cfg.Model.DefaultModel, Phase: "completed"}}
	if err = database.SaveWorkflowRun(&run); err != nil {
		t.Fatal(err)
	}
	if err = database.SaveDocumentOutput(session, run.ID, "output.txt", "text/plain; charset=utf-8", []byte("첫째 장\n\n앨리스는 흰 토끼를 따라 정원으로 들어갔다.\n\n둘째 장\n\n끝까지 저장한 번역문을 문서로 변환한다.")); err != nil {
		t.Fatal(err)
	}
	report := map[string]any{}
	for _, format := range []string{"docx", "pdf", "hwp", "hwpx"} {
		started := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
		err = s.convertDocumentTask(ctx, run, format)
		cancel()
		if err != nil {
			t.Fatal(format, err)
		}
		data, mime, err := database.DocumentOutput(session, run.ID, "output."+format)
		if err != nil {
			t.Fatal(err)
		}
		magic := []byte("PK")
		if format == "pdf" {
			magic = []byte("%PDF-")
		}
		if format == "hwp" {
			magic = []byte{0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1}
		}
		if !bytes.HasPrefix(data, magic) {
			t.Fatalf("invalid %s document", format)
		}
		if err = os.WriteFile(filepath.Join(dir, "output."+format), data, 0600); err != nil {
			t.Fatal(err)
		}
		report[format] = map[string]any{"bytes": len(data), "mime": mime, "seconds": time.Since(started).Seconds()}
		t.Log(format, len(data))
	}
	data, _ := json.MarshalIndent(report, "", "  ")
	if err = os.WriteFile(filepath.Join(dir, "conversion.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
}

func TestDocumentPDFExtractionLive(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_CONVERT_LIVE")
	if dir == "" {
		t.Skip("opt-in extraction from generated Korean text PDF")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "output.pdf"))
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{knowledgeIndex: &knowledge.Extractor{}}
	book, err := s.sourceFromBytes(context.Background(), raw, "application/pdf", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(book.Text, "앨리스는 흰 토끼를 따라 정원으로 들어갔다.") {
		t.Fatalf("embedded Korean PDF text missing: %q", book.Text)
	}
	data, _ := json.MarshalIndent(map[string]any{"embedded_korean_text_verified": true, "blocks": len(book.Blocks), "chars": len([]rune(book.Text))}, "", "  ")
	if err = os.WriteFile(filepath.Join(dir, "pdf-source.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
}
