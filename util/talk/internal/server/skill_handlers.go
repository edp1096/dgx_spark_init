package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"sparktalk/internal/skills"
	"strings"
)

var skillNamePattern = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,63}$`)
var skillToolsets = map[string]bool{"web": true, "media": true, "image": true, "ssh": true, "knowledge": true, "memory": true, "documents": true}

type publicSkill struct {
	skills.Skill
	Available bool   `json:"available"`
	Reason    string `json:"reason,omitempty"`
}

// Load a request snapshot; stored procedures never change midway through a request.
func (s *Server) allSkills() ([]skills.Skill, error) {
	items := skills.Catalog()
	active := map[string]bool{"web": true, "media": true, "image": true, "ssh": true}
	for i := range items {
		item, err := skills.Load(items[i].Name, active)
		if err != nil {
			return nil, err
		}
		items[i] = item
	}
	if s == nil || s.db == nil {
		return items, nil
	}
	settings, err := s.db.BuiltinSkillSettings()
	if err != nil {
		return nil, err
	}
	for i := range items {
		if value, ok := settings[items[i].Name]; ok {
			items[i].Enabled = value
		}
	}
	custom, err := s.db.Skills()
	if err != nil {
		return nil, err
	}
	return append(items, custom...), nil
}
func usableSkill(item skills.Skill, active map[string]bool) bool {
	if !item.Enabled {
		return false
	}
	for _, toolset := range item.Toolsets {
		if !active[toolset] {
			return false
		}
	}
	return true
}
func (s *Server) skillCatalog(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	items, err := s.allSkills()
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	cfg, _ := s.snapshot()
	active := map[string]bool{"web": cfg.Tools.Enabled, "media": cfg.Tools.MediaImportEnabled, "image": cfg.Image.Enabled, "ssh": false, "memory": cfg.Memory.Enabled, "documents": cfg.Extra.DocumentsEnabled}
	if s.db != nil {
		hosts, _ := s.db.SSHHosts()
		active["ssh"] = cfg.Extra.SSHEnabled && len(hosts) > 0
		count, _ := s.db.ReadyKnowledgeDocumentCount()
		collections, _ := s.db.KnowledgeCollections()
		active["knowledge"] = count > 0 || (len(collections) > 0 && cfg.Extra.CollectorEnabled && strings.TrimSpace(cfg.Extra.CollectorEndpoint) != "")
	}
	out := []publicSkill{}
	for _, item := range items {
		reason := ""
		if !cfg.Tools.SkillsEnabled {
			reason = "설정에서 스킬 사용이 꺼져 있습니다."
		} else if !item.Enabled {
			reason = "사용 안 함"
		} else {
			missing := []string{}
			for _, toolset := range item.Toolsets {
				if !active[toolset] {
					missing = append(missing, toolset)
				}
			}
			if len(missing) > 0 {
				reason = "필요한 도구: " + strings.Join(missing, ", ")
			}
		}
		item.Instructions = ""
		out = append(out, publicSkill{Skill: item, Available: reason == "", Reason: reason})
	}
	writeJSON(w, 200, out)
}
func (s *Server) skillItem(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimPrefix(r.URL.Path, "/api/skills/")
	if !skillNamePattern.MatchString(name) {
		http.Error(w, "이름은 영문 소문자·숫자·하이픈으로 1~64자 입력하세요.", 400)
		return
	}
	items, err := s.allSkills()
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	var existing *skills.Skill
	for i := range items {
		if items[i].Name == name {
			existing = &items[i]
			break
		}
	}
	if r.Method == http.MethodGet {
		if existing == nil {
			http.NotFound(w, r)
			return
		}
		writeJSON(w, 200, existing)
		return
	}
	if s.db == nil {
		http.Error(w, "저장소가 연결되지 않았습니다.", 503)
		return
	}
	switch r.Method {
	case http.MethodPut:
		var item skills.Skill
		decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 128<<10))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&item); err != nil {
			http.Error(w, "잘못된 스킬 데이터입니다.", 400)
			return
		}
		item.Name = name
		item.Description = strings.TrimSpace(item.Description)
		item.Instructions = strings.TrimSpace(item.Instructions)
		if existing != nil && existing.Builtin {
			// Built-ins are immutable. Only the enabled flag may be changed.
			if item.Description != "" || item.Instructions != "" || len(item.Toolsets) > 0 {
				http.Error(w, "내장 스킬은 복사한 뒤 수정하세요.", 400)
				return
			}
			err = s.db.SetBuiltinSkillEnabled(name, item.Enabled)
		} else {
			if item.Description == "" || len([]rune(item.Description)) > 500 || item.Instructions == "" || len([]rune(item.Instructions)) > 16000 {
				http.Error(w, "사용할 상황은 1~500자, 절차는 1~16,000자로 입력하세요.", 400)
				return
			}
			seen := map[string]bool{}
			normalized := []string{}
			for _, toolset := range item.Toolsets {
				if !skillToolsets[toolset] {
					http.Error(w, fmt.Sprintf("지원하지 않는 도구: %s", toolset), 400)
					return
				}
				if !seen[toolset] {
					normalized = append(normalized, toolset)
					seen[toolset] = true
				}
			}
			item.Toolsets = normalized
			item.Builtin = false
			err = s.db.SaveSkill(item)
		}
	case http.MethodDelete:
		if existing == nil {
			http.NotFound(w, r)
			return
		}
		if existing.Builtin {
			http.Error(w, "내장 스킬은 삭제 대신 사용을 끌 수 있습니다.", 400)
			return
		}
		err = s.db.DeleteSkill(name)
	default:
		methodNotAllowed(w)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
