package server

import (
	"path/filepath"
	"time"
)

// ResumeInterruptedJobs restores durable queued work. Running generators are
// failed rather than submitted twice because their remote request state is not
// durable; users can explicitly retry them.
func (s *Server) ResumeInterruptedJobs() (resumed, failed int) {
	generationResumed := false
	subtitleResumed := false
	for _, persisted := range s.jobs.List() {
		if persisted.Status != "queued" && persisted.Status != "running" {
			continue
		}
		if isGenerationKind(persisted.Kind) {
			if persisted.Status == "running" {
				persisted.Status = "failed"
				persisted.Error = "앱 재시작으로 실행 중이던 작업이 중단되었습니다. 재시도할 수 있습니다."
				_ = s.jobs.Save(persisted)
				failed++
				continue
			}
			if persisted.Params == nil {
				persisted.Params = map[string]any{}
			}
			if _, ok := persisted.Params["queued_at"]; !ok {
				persisted.Params["queued_at"] = persisted.CreatedAt.Format(time.RFC3339Nano)
			}
			persisted.Params["stage"] = "queued"
			persisted.Error = ""
			_ = s.jobs.Save(persisted)
			resumed++
			generationResumed = true
			continue
		}
		if persisted.Kind != "recognition" {
			persisted.Status = "failed"
			persisted.Error = "앱 재시작으로 작업이 중단되었습니다."
			_ = s.jobs.Save(persisted)
			failed++
			continue
		}

		inputDir := filepath.Join(s.dataDir, "inputs", persisted.ID)
		sourceKind := decodeSubtitleJobParams(persisted.Params, s.config().Recognition).Source
		if sourceKind != "url" {
			matches, _ := filepath.Glob(filepath.Join(inputDir, "source.*"))
			if len(matches) == 0 {
				persisted.Status = "failed"
				persisted.Error = "앱 재시작 후 원본 입력 파일을 찾을 수 없습니다."
				_ = s.jobs.Save(persisted)
				failed++
				continue
			}
		}

		persisted.Status = "queued"
		persisted.Error = ""
		if persisted.Params == nil {
			persisted.Params = map[string]any{}
		}
		if _, ok := persisted.Params["queued_at"]; !ok {
			persisted.Params["queued_at"] = persisted.CreatedAt.Format(time.RFC3339Nano)
		}
		persisted.Params["stage"] = "queued"
		_ = s.jobs.Save(persisted)
		resumed++
		subtitleResumed = true
	}
	if subtitleResumed {
		s.wakeSubtitleQueue()
	}
	if generationResumed {
		s.wakeGenerationQueue()
	}
	return resumed, failed
}
