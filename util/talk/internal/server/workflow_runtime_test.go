package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/skills"
	"sparktalk/internal/workflows"
	"strings"
	"testing"
)

func workflowFixture(t *testing.T) (*Server, config.ToolsConfig) {
	t.Helper()
	store, e := db.Open(t.TempDir() + "/test.db")
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { store.Close() })
	store.CreateSession("s", "test", "model", "none")
	cfg := config.ToolsConfig{SkillsEnabled: true, MaxRounds: 6}
	s := &Server{db: store, cfg: config.Config{Tools: cfg}}
	for _, name := range []string{"first", "second"} {
		if e = store.SaveSkill(skills.Skill{Name: name, Description: name, Instructions: "BODY " + name, Enabled: true, Toolsets: []string{}}); e != nil {
			t.Fatal(e)
		}
	}
	return s, cfg
}
func workflowDefinition() workflows.Definition {
	return workflows.Definition{Name: "fixture", Description: "test", Enabled: true, Steps: []workflows.Step{{Name: "first", Skills: []string{"first"}, Goal: "first goal", DoneWhen: "first output", OnFailure: -1}, {Name: "second", Skills: []string{"second"}, Goal: "second goal", DoneWhen: "second output", OnFailure: -1}}}
}
func writeReport(w http.ResponseWriter, status, summary string) {
	report, _ := json.Marshal(workflows.Report{Status: status, Summary: summary, Evidence: []string{"fabricated"}})
	chunk, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []any{map[string]any{"index": 0, "id": "report", "type": "function", "function": map[string]any{"name": "workflow_report", "arguments": string(report)}}}}, "finish_reason": "tool_calls"}}})
	fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", chunk)
}
func TestWorkflowSnapshotsStagesPausesAndResumes(t *testing.T) {
	s, cfg := workflowFixture(t)
	s.db.SaveWorkflow(workflowDefinition())
	calls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var body struct {
			Messages []llm.Message `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		raw, _ := json.Marshal(body.Messages)
		text := string(raw)
		if calls == 1 {
			if !strings.Contains(text, "BODY first") || strings.Contains(text, "BODY second") {
				t.Error("loaded future instructions")
			}
			s.db.SaveSkill(skills.Skill{Name: "second", Description: "second", Instructions: "CHANGED BODY", Enabled: true})
			writeReport(w, "completed", "first deliverable")
			return
		}
		if !strings.Contains(text, "BODY second") || strings.Contains(text, "CHANGED BODY") || strings.Contains(text, "BODY first") {
			t.Error("snapshot/current-stage isolation failed")
		}
		if !strings.Contains(text, "first deliverable") {
			t.Error("handoff lost")
		}
		if calls == 2 {
			writeReport(w, "blocked", "need input")
		} else {
			writeReport(w, "completed", "final deliverable")
		}
	}))
	defer backend.Close()
	client := llm.New(backend.URL, "model", "")
	invoke := func(marker string) (completionResult, error) {
		return runCompletionLoopForSession(s, "s", context.Background(), client, []llm.Message{{Role: "user", Content: marker + "\nwrite document"}}, "model", "none", "", cfg, false, func(string, any) error { return nil })
	}
	if _, e := invoke("@workflow:fixture"); e == nil {
		t.Fatal("blocked step silently completed")
	}
	runs, e := s.db.WorkflowRuns("s")
	if e != nil || len(runs) != 1 || runs[0].Current != 1 || runs[0].Status != "paused" {
		t.Fatalf("pause lost: %+v %v", runs, e)
	}
	result, e := invoke("@resume:" + runs[0].ID)
	if e != nil || result.Content != "final deliverable" || calls != 3 {
		t.Fatalf("resume reran completed stages: %+v %v calls=%d", result, e, calls)
	}
	runs, _ = s.db.WorkflowRuns("s")
	if runs[0].Status != "completed" || len(runs[0].Steps[1].History) != 1 {
		t.Fatal("final state or previous attempt lost")
	}
	if _, e = s.db.ClaimWorkflowRun(runs[0].ID, "another-session"); e == nil {
		t.Fatal("cross-session resume allowed")
	}
}
func TestWorkflowCannotInventVerificationEvidence(t *testing.T) {
	s, cfg := workflowFixture(t)
	def := workflowDefinition()
	def.Steps = def.Steps[:1]
	def.Steps[0].VerifyTool = "ssh_exec"
	s.db.SaveWorkflow(def)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { writeReport(w, "completed", "tests passed") }))
	defer backend.Close()
	_, e := runCompletionLoopForSession(s, "s", context.Background(), llm.New(backend.URL, "model", ""), []llm.Message{{Role: "user", Content: "@workflow:fixture\ntest"}}, "model", "none", "", cfg, false, func(string, any) error { return nil })
	if e == nil {
		t.Fatal("fabricated evidence accepted")
	}
	runs, _ := s.db.WorkflowRuns("s")
	if runs[0].Steps[0].Status != "unverified" {
		t.Fatal(runs[0].Steps[0].Status)
	}
}
func TestWorkflowRetryIsBounded(t *testing.T) {
	s, cfg := workflowFixture(t)
	def := workflowDefinition()
	def.Steps[1].OnFailure = 0
	def.Steps[1].MaxRetries = 1
	s.db.SaveWorkflow(def)
	calls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls%2 == 1 {
			writeReport(w, "completed", "draft")
		} else {
			writeReport(w, "failed", "test failed")
		}
	}))
	defer backend.Close()
	_, e := runCompletionLoopForSession(s, "s", context.Background(), llm.New(backend.URL, "model", ""), []llm.Message{{Role: "user", Content: "@workflow:fixture\ntest"}}, "model", "none", "", cfg, false, func(string, any) error { return nil })
	if e == nil || calls != 4 {
		t.Fatalf("unbounded/absent retry: %d %v", calls, e)
	}
}
func TestWorkflowEvidenceChecksExitAndCommand(t *testing.T) {
	step := workflows.Step{VerifyTool: "ssh_exec", VerifyCommand: "go test ./..."}
	proof := workflows.Evidence{Tool: "ssh_exec", Arguments: `{"command":"go test ./..."}`, Result: `{"exit_code":0,"stdout":"ok"}`}
	if !evidenceMatches(step, proof) {
		t.Fatal("valid proof rejected")
	}
	for _, out := range []string{`{"exit_code":1}`, `{"stdout":"passed"}`, `{"exit_code":0,"error":"failure"}`, `not json`} {
		proof.Result = out
		if evidenceMatches(step, proof) {
			t.Fatal("invalid proof accepted", out)
		}
	}
	proof.Result = `{"exit_code":0}`
	proof.Arguments = `{"command":"echo go test ./..."}`
	if evidenceMatches(step, proof) {
		t.Fatal("different command accepted")
	}
}

func TestWorkflowAutomaticStartAndManualSkillOverride(t *testing.T) {
	for _, manual := range []bool{false, true} {
		t.Run(fmt.Sprint(manual), func(t *testing.T) {
			s, cfg := workflowFixture(t)
			def := workflowDefinition()
			def.Steps = def.Steps[:1]
			if e := s.db.SaveWorkflow(def); e != nil {
				t.Fatal(e)
			}
			calls := 0
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				var body struct {
					Tools []llm.Tool `json:"tools"`
				}
				json.NewDecoder(r.Body).Decode(&body)
				available := false
				for _, tool := range body.Tools {
					if tool.Function.Name == "workflow_start" {
						available = true
					}
				}
				if calls == 1 {
					if available == manual {
						t.Error("automatic workflow selection ignored manual skill preference")
					}
					if manual {
						fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"manual\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
						return
					}
					fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"start\",\"type\":\"function\",\"function\":{\"name\":\"workflow_start\",\"arguments\":\"{\\\"name\\\":\\\"fixture\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n")
					return
				}
				if available {
					t.Error("nested workflow selection exposed")
				}
				writeReport(w, "completed", "done")
			}))
			defer backend.Close()
			input := "write document"
			if manual {
				input = "@skill:first\n" + input
			}
			_, err := runCompletionLoopForSession(s, "s", context.Background(), llm.New(backend.URL, "model", ""), []llm.Message{{Role: "user", Content: input}}, "model", "none", "", cfg, false, func(string, any) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
			runs, err := s.db.WorkflowRuns("s")
			if err != nil {
				t.Fatal(err)
			}
			if manual && len(runs) != 0 {
				t.Fatal("manual skill unexpectedly started workflow")
			}
			if !manual && (len(runs) != 1 || runs[0].Goal != "write document" || runs[0].Status != "completed") {
				t.Fatalf("automatic dispatch lost task: %+v", runs)
			}
		})
	}
}

func TestWorkflowReportAfterLastToolRound(t *testing.T) {
	s, cfg := workflowFixture(t)
	cfg.MaxRounds = 1
	def := workflowDefinition()
	def.Steps = def.Steps[:1]
	s.db.SaveWorkflow(def)
	calls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var body struct {
			Tools []llm.Tool `json:"tools"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		if calls == 1 {
			// Loading an already selected skill must not consume the single work round.
			fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"load","type":"function","function":{"name":"skill_view","arguments":"{\"name\":\"first\"}"}}]},"finish_reason":"tool_calls"}]}`+"\n\ndata: [DONE]\n\n")
			return
		}
		if calls == 2 {
			if len(body.Tools) <= 1 {
				t.Error("skill loading consumed work budget")
			}
			fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"work","type":"function","function":{"name":"unknown_tool","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`+"\n\ndata: [DONE]\n\n")
			return
		}
		runs, e := s.db.WorkflowRuns("s")
		if e != nil || len(runs) != 1 || len(runs[0].Steps[0].Evidence) != 1 {
			t.Error("tool evidence was not checkpointed before next request")
		}
		if calls != 3 || len(body.Tools) != 1 || body.Tools[0].Function.Name != "workflow_report" {
			t.Errorf("last round cannot report: %d %+v", calls, body.Tools)
		}
		writeReport(w, "blocked", "tool unavailable")
	}))
	defer backend.Close()
	result, err := runCompletionLoopForSession(s, "s", context.Background(), llm.New(backend.URL, "model", ""), []llm.Message{{Role: "user", Content: "@workflow:fixture\nwork"}}, "model", "none", "", cfg, false, func(string, any) error { return nil })
	if err == nil || calls != 3 || result.Content != "tool unavailable" {
		t.Fatalf("report lost: %+v %v calls=%d", result, err, calls)
	}
	runs, _ := s.db.WorkflowRuns("s")
	if runs[0].Steps[0].Status != "blocked" || len(runs[0].Steps[0].Evidence) != 1 || runs[0].Steps[0].Evidence[0].Error == "" {
		t.Fatal("failed actual tool result lost")
	}
}

func TestWorkflowValidationRejectsInvalidDependencies(t *testing.T) {
	s, _ := workflowFixture(t)
	for _, def := range workflows.Defaults() {
		if e := s.validateWorkflow(def); e != nil {
			t.Fatalf("builtin %s: %v", def.Name, e)
		}
	}
	for _, change := range []func(*workflows.Definition){
		func(d *workflows.Definition) { d.Steps[1].OnFailure = 1 },
		func(d *workflows.Definition) { d.Steps[1].MaxRetries = 3 },
		func(d *workflows.Definition) { d.Steps[0].Skills = []string{"missing"} },
		func(d *workflows.Definition) { d.Steps[0].VerifyCommand = "go test ./..." },
		func(d *workflows.Definition) { d.Steps[0].VerifyTool = "arbitrary" },
	} {
		def := workflowDefinition()
		change(&def)
		if e := s.validateWorkflow(def); e == nil {
			t.Fatal("invalid procedure accepted")
		}
	}
}
