package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"sparktalk/internal/workflows"
	"strings"
)

var workflowEvidenceTools = map[string]bool{"": true, "ssh_exec": true, "web_search": true, "web_fetch": true, "web_collect": true, "media_import": true, "image_generate": true, "knowledge_search": true, "document_generate": true}

func (s *Server) allWorkflows() ([]workflows.Definition, error) {
	items := workflows.Defaults()
	if s == nil || s.db == nil {
		return items, nil
	}
	custom, e := s.db.Workflows()
	if e != nil {
		return nil, e
	}
	by := map[string]workflows.Definition{}
	for _, x := range items {
		by[x.Name] = x
	}
	for _, x := range custom {
		by[x.Name] = x
	}
	items = nil
	for _, x := range by {
		items = append(items, x)
	}
	sort.Slice(items, func(i, j int) bool { return items[i].Name < items[j].Name })
	return items, nil
}
func (s *Server) validateWorkflow(x workflows.Definition) error {
	if !skillNamePattern.MatchString(x.Name) || strings.TrimSpace(x.Description) == "" || len([]rune(x.Description)) > 500 || len(x.Steps) < 1 || len(x.Steps) > 8 {
		return fmt.Errorf("이름·설명을 입력하고 단계를 1~8개 구성하세요")
	}
	items, e := s.allSkills()
	if e != nil {
		return e
	}
	known := map[string]bool{}
	for _, v := range items {
		known[v.Name] = true
	}
	for i, step := range x.Steps {
		if strings.TrimSpace(step.Name) == "" || len([]rune(step.Name)) > 80 || strings.TrimSpace(step.Goal) == "" || strings.TrimSpace(step.DoneWhen) == "" || len([]rune(step.Goal)) > 4000 || len([]rune(step.DoneWhen)) > 2000 || len(step.Skills) < 1 || len(step.Skills) > 4 || !workflowEvidenceTools[step.VerifyTool] || len(step.VerifyCommand) > 2000 || (step.VerifyCommand != "" && step.VerifyTool != "ssh_exec") || step.OnFailure < -1 || step.OnFailure >= i || step.MaxRetries < 0 || step.MaxRetries > 2 {
			return fmt.Errorf("%d단계의 목표·완료 조건·검증 설정을 확인하세요", i+1)
		}
		for _, name := range step.Skills {
			if !known[name] {
				return fmt.Errorf("없는 스킬: %s", name)
			}
		}
	}
	return nil
}
func (s *Server) workflowCatalog(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		methodNotAllowed(w)
		return
	}
	items, e := s.allWorkflows()
	if e != nil {
		http.Error(w, e.Error(), 500)
		return
	}
	if r.URL.Query().Get("selection") == "1" {
		cfg, _ := s.snapshot()
		registry := newCompletionToolRegistry(s, "", cfg.Tools, cfg.Tools.Enabled, nil)
		if registry.err != nil {
			http.Error(w, registry.err.Error(), 500)
			return
		}
		type choice struct {
			workflows.Definition
			Missing []string `json:"missing"`
		}
		out := []choice{}
		for _, item := range items {
			missing := []string{}
			for _, step := range item.Steps {
				if step.VerifyTool != "" {
					if _, ok := registry.handlers[step.VerifyTool]; !ok {
						missing = append(missing, step.VerifyTool)
					}
				}
			}
			out = append(out, choice{item, missing})
		}
		writeJSON(w, 200, out)
		return
	}
	writeJSON(w, 200, items)
}
func (s *Server) workflowItem(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimPrefix(r.URL.Path, "/api/workflows/")
	items, e := s.allWorkflows()
	if e != nil {
		http.Error(w, e.Error(), 500)
		return
	}
	var existing *workflows.Definition
	for i := range items {
		if items[i].Name == name {
			existing = &items[i]
			break
		}
	}
	switch r.Method {
	case "GET":
		if existing == nil {
			http.NotFound(w, r)
			return
		}
		writeJSON(w, 200, existing)
		return
	case "PUT":
		var x workflows.Definition
		dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 128<<10))
		dec.DisallowUnknownFields()
		if e = dec.Decode(&x); e != nil {
			http.Error(w, "잘못된 작업 절차입니다", 400)
			return
		}
		x.Name = name
		if existing != nil && existing.Builtin {
			if x.Description != "" || len(x.Steps) > 0 {
				http.Error(w, "내장 절차는 복사해서 수정하세요", 400)
				return
			}
			enabled := x.Enabled
			x = *existing
			x.Enabled = enabled
		} else {
			x.Builtin = false
			if e = s.validateWorkflow(x); e != nil {
				http.Error(w, e.Error(), 400)
				return
			}
		}
		e = s.db.SaveWorkflow(x)
	case "DELETE":
		if existing == nil {
			http.NotFound(w, r)
			return
		}
		if existing.Builtin {
			http.Error(w, "내장 절차는 사용을 끌 수 있습니다", 400)
			return
		}
		e = s.db.DeleteWorkflow(name)
	default:
		methodNotAllowed(w)
		return
	}
	if e != nil {
		http.Error(w, e.Error(), 500)
		return
	}
	w.WriteHeader(204)
}
func (s *Server) workflowRuns(w http.ResponseWriter, r *http.Request, session string) {
	if r.Method != "GET" {
		methodNotAllowed(w)
		return
	}
	items, e := s.db.WorkflowRuns(session)
	if e != nil {
		http.Error(w, e.Error(), 500)
		return
	}
	for i := range items {
		items[i].Skills = nil
	}
	writeJSON(w, 200, items)
}
