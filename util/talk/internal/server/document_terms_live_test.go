package server

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/llm"
	"sparktalk/internal/performance"
	"sparktalk/internal/workflows"
)

// Isolated source/target replay; no production DB writes or model/set changes.
func TestDocumentTermsQualificationLive(t *testing.T) {
	dir := os.Getenv("SPARKTALK_TERMS_LIVE")
	if dir == "" {
		t.Skip("explicit isolated terms qualification")
	}
	var run workflows.Run
	var units []workflows.DocumentUnit
	read := func(name string, v any) {
		raw, e := os.ReadFile(filepath.Join(dir, name))
		if e != nil {
			t.Fatal(e)
		}
		if e = json.Unmarshal(raw, v); e != nil {
			t.Fatal(e)
		}
	}
	read("alice-run.json", &run)
	read("alice-units.json", &units)
	db, e := sql.Open("sqlite", "file:/home/edp1096/workspace/dgx_spark_init/util/talk/dist/sparktalk.db?mode=ro")
	if e != nil {
		t.Fatal(e)
	}
	defer db.Close()
	idle := func() bool {
		var n int
		e := db.QueryRow(`SELECT (SELECT count(*) FROM messages WHERE status NOT IN ('completed','cancelled','failed'))+(SELECT count(*) FROM workflow_runs WHERE status='running')`).Scan(&n)
		return e == nil && n == 0
	}
	if !idle() {
		t.Fatal("production work active")
	}
	resp, e := http.Get("http://127.0.0.1:8585/api/config")
	if e != nil {
		t.Fatal(e)
	}
	var cfg config.Config
	e = json.NewDecoder(resp.Body).Decode(&cfg)
	resp.Body.Close()
	if e != nil {
		t.Fatal(e)
	}
	if cfg.Model.DefaultModel != "qwen38fn_exl3" {
		t.Fatal("unexpected model")
	}
	run.Document.Pipeline = workflows.DocumentTermsVersion
	run.Document.Effort = "medium"
	run.Document.Terms = &workflows.DocumentTerms{Repair: true}
	for source := range run.Document.Glossary {
		run.Document.Terms.Candidates = append(run.Document.Terms.Candidates, workflows.DocumentEntity{Source: source})
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()
	go func() {
		tick := time.NewTicker(time.Second)
		defer tick.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
				if !idle() {
					cancel()
					return
				}
			}
		}
	}()
	client := llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType).WithBackendCancellation(ctx)
	call := func(name string, request documentStageRequest) llm.StreamResult {
		var memMu sync.Mutex
		memory := []map[string]float64{}
		observe := func() {
			raw, _ := os.ReadFile("/proc/meminfo")
			row := map[string]float64{"time": float64(time.Now().UnixMilli()) / 1000}
			for _, line := range strings.Split(string(raw), "\n") {
				parts := strings.Fields(line)
				if len(parts) > 1 {
					key := strings.TrimSuffix(parts[0], ":")
					if key == "MemAvailable" || key == "MemFree" || key == "SwapFree" || key == "SwapTotal" {
						v, _ := strconv.ParseFloat(parts[1], 64)
						row[key] = v / 1048576
					}
				}
			}
			memMu.Lock()
			memory = append(memory, row)
			memMu.Unlock()
		}
		done := make(chan struct{})
		stopped := make(chan struct{})
		observe()
		go func() {
			defer close(stopped)
			timer := time.NewTicker(200 * time.Millisecond)
			defer timer.Stop()
			for {
				select {
				case <-done:
					return
				case <-timer.C:
					observe()
				}
			}
		}()
		var perf performance.Accumulator
		requestCtx := llm.WithPerformanceObserver(llm.WithTemperature(ctx, .2), perf.Start)
		st := time.Now()
		result, e := request(requestCtx, client, run, workflows.DocumentUnit{}, cfg.Context.WindowTokens, func(string, string) error { return ctx.Err() })
		close(done)
		<-stopped
		observe()
		payload := map[string]any{"memory": memory, "model": cfg.Model.DefaultModel, "effort": "medium", "seconds": time.Since(st).Seconds(), "performance": perf.Summary(false), "result": result}
		if e != nil {
			payload["error"] = e.Error()
		}
		raw, _ := json.MarshalIndent(payload, "", "  ")
		os.WriteFile(filepath.Join(dir, name+".json"), raw, 0600)
		if e != nil {
			t.Fatal(e)
		}
		t.Log(name, time.Since(st))
		return result
	}
	wanted := os.Getenv("SPARKTALK_TERMS_SOURCE")
	for _, c := range documentTermCases(run, units) {
		if wanted != "" && c.Source != wanted {
			continue
		}
		for start := 0; start < len(c.References); {
			end := start
			chars := 0
			for end < len(c.References) && end-start < 8 {
				n := len([]rune(c.References[end].Source + c.References[end].Target))
				if end > start && chars+n > 8000 {
					break
				}
				chars += n
				end++
			}
			refs := c.References[start:end]
			result := call(fmt.Sprintf("extract-%s-%d", c.Source, start), extractTermRequest(c.Source, refs))
			invalid := validateTermExtraction(&c, refs, result)
			if len(invalid) > 0 {
				t.Fatalf("%s extraction: %v", c.Source, invalid)
			}
			start = end
		}
		result := call("decide-"+c.Source, decideTermRequest(c))
		if e := validateTermDecision(&c, result); e != nil {
			t.Fatal(e)
		}
		for gi, g := range c.Decisions {
			if g.Status != "correctable" {
				continue
			}
			refs := []workflows.DocumentTermReference{}
			for _, r := range c.References {
				for _, id := range g.References {
					if r.ID == id {
						refs = append(refs, r)
					}
				}
			}
			for start := 0; start < len(refs); start += 4 {
				batch := refs[start:min(len(refs), start+4)]
				result := call(fmt.Sprintf("patch-%s-%d-%d", c.Source, gi, start), patchTermRequest(c.Source, g.Canonical, batch))
				var payload struct {
					Patches map[string][]workflows.DocumentTermPatch `json:"patches"`
				}
				if e := decodeTermJSON(result, &payload); e != nil {
					t.Fatal(e)
				}
				proposed := map[string]string{}
				for i, r := range batch {
					p, ok := payload.Patches[r.ID]
					if !ok {
						t.Fatal("missing patch")
					}
					_, v, e := applyTermPatches(units, r, g.Canonical, p)
					if e != nil {
						t.Fatalf("%s %s: %v", c.Source, r.ID, e)
					}
					proposed[r.ID] = v
					batch[i].Patches = p
				}
				call(fmt.Sprintf("verify-%s-%d-%d", c.Source, gi, start), verifyTermRequest(c.Source, batch, proposed))
			}
		}
		raw, _ := json.MarshalIndent(c, "", "  ")
		os.WriteFile(filepath.Join(dir, "case-"+c.Source+".json"), raw, 0600)
	}
}

// Recheck an archived model response after parser changes without rerunning a
// model or altering any production checkpoint.
func TestDocumentTermsRecordedExtraction(t *testing.T) {
	path := os.Getenv("SPARKTALK_TERMS_EXTRACTION_REPLAY")
	if path == "" {
		t.Skip("explicit archived extraction replay")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var sample struct {
		Case   workflows.DocumentTermCase `json:"case"`
		Result llm.StreamResult           `json:"result"`
	}
	if err = json.Unmarshal(raw, &sample); err != nil {
		t.Fatal(err)
	}
	sample.Result.FinishReason = "stop"
	if failures := validateTermExtraction(&sample.Case, sample.Case.References, sample.Result); len(failures) > 0 {
		t.Fatal(failures)
	}
	for _, r := range sample.Case.References {
		if r.Status != "named" {
			t.Fatal(r.Status)
		}
	}
	t.Log("original model extraction accepted; source/target and production records unchanged")
}
