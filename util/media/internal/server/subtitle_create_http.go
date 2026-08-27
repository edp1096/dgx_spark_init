package server

import (
	"fmt"
	"io"
	"mediaapp/internal/jobs"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func (s *Server) createSubtitle(w http.ResponseWriter, r *http.Request) {
	cfg := s.config()
	maxBytes := cfg.Recognition.MaxUploadMB << 20
	r.Body = http.MaxBytesReader(w, r.Body, maxBytes+(2<<20))
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		http.Error(w, "invalid or oversized form", http.StatusBadRequest)
		return
	}
	if r.MultipartForm != nil {
		defer r.MultipartForm.RemoveAll()
	}

	file, header, fileErr := r.FormFile("media")
	if fileErr != nil {
		file, header, fileErr = r.FormFile("audio") // 이전 클라이언트 호환
	}
	if fileErr == nil {
		defer file.Close()
	}
	sourceURL := strings.TrimSpace(r.FormValue("url"))
	reuseVideoID := strings.TrimSpace(r.FormValue("reuse_video_job"))
	sourceCount := 0
	if fileErr == nil {
		sourceCount++
	}
	if sourceURL != "" {
		sourceCount++
	}
	if reuseVideoID != "" {
		sourceCount++
	}
	if sourceCount != 1 {
		http.Error(w, "파일, 링크 또는 생성 영상 중 하나만 선택하세요", http.StatusBadRequest)
		return
	}
	var reusedVideo jobs.Job
	if reuseVideoID != "" {
		var ok bool
		reusedVideo, ok = s.jobs.Get(reuseVideoID)
		if !ok || reusedVideo.Kind != "video" || reusedVideo.Status != "completed" || reusedVideo.OutputURL == "" {
			http.Error(w, "선택한 생성 영상을 사용할 수 없습니다", http.StatusConflict)
			return
		}
	}

	id := newID()
	inputDir := filepath.Join(s.dataDir, "inputs", id)
	if err := os.MkdirAll(inputDir, 0o755); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	inputPath := ""
	sourceName := sourceURL
	if fileErr == nil {
		ext := strings.ToLower(filepath.Ext(header.Filename))
		if ext == "" {
			ext = ".media"
		}
		inputPath = filepath.Join(inputDir, "source"+ext)
		destination, err := os.Create(inputPath)
		if err == nil {
			var written int64
			written, err = io.Copy(destination, io.LimitReader(file, maxBytes+1))
			closeErr := destination.Close()
			if err == nil {
				err = closeErr
			}
			if err == nil && written > maxBytes {
				err = fmt.Errorf("파일이 너무 큽니다 (최대 %d MB)", cfg.Recognition.MaxUploadMB)
			}
		}
		if err != nil {
			_ = os.RemoveAll(inputDir)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		sourceName = header.Filename
	} else if reuseVideoID != "" {
		source := s.jobs.OutputPath(filepath.Base(reusedVideo.OutputURL))
		if _, err := os.Stat(source); err != nil {
			_ = os.RemoveAll(inputDir)
			http.Error(w, "선택한 생성 영상 파일이 더 이상 없습니다", http.StatusNotFound)
			return
		}
		ext := strings.ToLower(filepath.Ext(source))
		if ext == "" {
			ext = ".mp4"
		}
		inputPath = filepath.Join(inputDir, "source"+ext)
		if err := linkOrCopyFile(source, inputPath); err != nil {
			_ = os.RemoveAll(inputDir)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		sourceName = "생성 영상 " + reusedVideo.ID[:8]
	}

	language := valueOr(r.FormValue("language"), cfg.Recognition.DefaultLanguage)
	context := strings.TrimSpace(r.FormValue("context"))
	formats, err := parseOutputFormats(r.FormValue("output_formats"), cfg.Recognition.DefaultOutputFormats)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	translationMode := valueOr(r.FormValue("translation_mode"), cfg.Recognition.DefaultTranslationMode)
	if translationMode != "none" && translationMode != "translated" && translationMode != "bilingual" {
		http.Error(w, "invalid translation_mode", http.StatusBadRequest)
		return
	}
	targetLanguage := valueOr(r.FormValue("target_language"), cfg.Recognition.DefaultTranslationLanguage)
	mediaPart := strings.TrimSpace(r.FormValue("media_part"))
	mediaSource := strings.TrimSpace(r.FormValue("media_source"))
	if len(mediaPart) > 32 || len(mediaSource) > 32 {
		http.Error(w, "invalid media part/source selection", http.StatusBadRequest)
		return
	}
	sourceKind := "file"
	if sourceURL != "" {
		sourceKind = "url"
	} else if reuseVideoID != "" {
		sourceKind = "video_job"
	}
	params := subtitleJobParams{
		Language: language, Context: context, Source: sourceKind, SourceJobID: reuseVideoID,
		OutputFormats: formats, TranslationMode: translationMode, TargetLanguage: targetLanguage,
		MediaPart: mediaPart, MediaSource: mediaSource, Stage: "queued",
	}
	now := time.Now()
	params.QueuedAt = now.Format(time.RFC3339Nano)
	j := jobs.Job{ID: id, Kind: "recognition", Status: "queued", Prompt: sourceName, Params: params.toMap(), CreatedAt: now}
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.wakeSubtitleQueue()
	writeJSON(w, http.StatusAccepted, j)
}

func (s *Server) retrySubtitle(w http.ResponseWriter, r *http.Request) {
	j, ok := s.jobs.Get(r.PathValue("id"))
	if !ok {
		http.NotFound(w, r)
		return
	}
	if j.Kind != "recognition" || (j.Status != "failed" && j.Status != "cancelled") {
		http.Error(w, "only failed or cancelled subtitle jobs can be resumed", http.StatusConflict)
		return
	}
	inputDir := filepath.Join(s.dataDir, "inputs", j.ID)
	params := decodeSubtitleJobParams(j.Params, s.config().Recognition)
	if params.Source != "url" {
		matches, _ := filepath.Glob(filepath.Join(inputDir, "source.*"))
		if len(matches) == 0 {
			http.Error(w, "saved source media is missing", http.StatusConflict)
			return
		}
	}
	// A failed client request does not necessarily mean the remote downloader
	// has exited. Stop and join it before reusing the durable request ID.
	s.cancelMediaPreparation(j.ID)

	workerJob := j
	workerJob.Status = "queued"
	workerJob.Error = ""
	workerJob.OutputURL = ""
	workerJob.Outputs = nil
	workerJob.CaptionURL = ""
	workerJob.Params = make(map[string]any, len(j.Params))
	for key, value := range j.Params {
		workerJob.Params[key] = value
	}
	for _, key := range []string{
		"media_downloaded_bytes", "media_total_bytes", "media_percent", "media_eta_seconds",
		"current_segment", "recognized_segments", "text", "cues", "started_at", "stage_started_at",
	} {
		delete(workerJob.Params, key)
	}
	workerJob.Params["stage"] = "queued"
	workerJob.Params["queued_at"] = time.Now().Format(time.RFC3339Nano)
	if err := s.jobs.Save(workerJob); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	s.wakeSubtitleQueue()
	writeJSON(w, http.StatusAccepted, workerJob)
}

func parseOutputFormats(value string, defaults []string) ([]string, error) {
	if strings.TrimSpace(value) == "" {
		return append([]string(nil), defaults...), nil
	}
	allowed := map[string]bool{"srt": true, "vtt": true, "timestamped_txt": true, "txt": true}
	seen := map[string]bool{}
	result := []string{}
	for _, item := range strings.Split(value, ",") {
		item = strings.TrimSpace(item)
		if !allowed[item] {
			return nil, fmt.Errorf("unsupported output format: %s", item)
		}
		if !seen[item] {
			seen[item] = true
			result = append(result, item)
		}
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("at least one output format is required")
	}
	return result, nil
}
