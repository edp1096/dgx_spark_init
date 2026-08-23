package server

import (
	"archive/zip"
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"mediaapp/internal/jobs"
)

var errJobCancelled = errors.New("job cancelled")

const autoMultilingualLanguage = "AutoMultilingual"

type preparedManifest struct {
	SourceName string            `json:"source_name"`
	Segments   []preparedSegment `json:"segments"`
	Asset      *preparedAsset    `json:"asset,omitempty"`
}

type preparedAsset struct {
	ID          string  `json:"id"`
	Filename    string  `json:"filename"`
	MediaType   string  `json:"media_type"`
	ContentType string  `json:"content_type"`
	Size        int64   `json:"size"`
	Duration    float64 `json:"duration"`
	Width       int     `json:"width"`
	Height      int     `json:"height"`
}

type preparedSegment struct {
	Name     string  `json:"name"`
	Start    float64 `json:"start"`
	End      float64 `json:"end"`
	Duration float64 `json:"duration"`
}

type timedWord struct {
	Text  string  `json:"text"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

type subtitleCue struct {
	Start      float64
	End        float64
	Text       string
	Translated string
}

type mediaProgressStatus struct {
	Stage           string  `json:"stage"`
	DownloadedBytes int64   `json:"downloaded_bytes"`
	TotalBytes      int64   `json:"total_bytes"`
	Percent         float64 `json:"percent"`
	ETASeconds      int     `json:"eta_seconds"`
}

func (s *Server) mediaOptions(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	if err := r.ParseForm(); err != nil {
		http.Error(w, "invalid form", http.StatusBadRequest)
		return
	}
	sourceURL := strings.TrimSpace(r.FormValue("url"))
	if sourceURL == "" {
		http.Error(w, "url is required", http.StatusBadRequest)
		return
	}
	endpoint := strings.TrimRight(s.config().Engines["media"].Endpoint, "/") + "/v1/media/options"
	data, contentType, err := s.callMultipart(endpoint, map[string]string{"url": sourceURL}, "", nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", valueOr(contentType, "application/json"))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

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
	if (fileErr == nil) == (sourceURL != "") {
		http.Error(w, "파일 또는 링크 중 하나만 입력하세요", http.StatusBadRequest)
		return
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
	params := map[string]any{
		"language": language, "context": context, "source": map[bool]string{true: "url", false: "file"}[sourceURL != ""],
		"output_formats": formats, "translation_mode": translationMode, "target_language": targetLanguage,
	}
	if mediaPart != "" {
		params["media_part"] = mediaPart
	}
	if mediaSource != "" {
		params["media_source"] = mediaSource
	}
	j := jobs.Job{ID: id, Kind: "recognition", Status: "queued", Prompt: sourceName, Params: params, CreatedAt: time.Now()}
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	workerJob := j
	workerJob.Params = make(map[string]any, len(j.Params))
	for key, value := range j.Params {
		workerJob.Params[key] = value
	}
	go s.runSubtitle(workerJob, inputDir, inputPath, sourceURL, language, context, formats, translationMode, targetLanguage, mediaPart, mediaSource)
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
	for _, active := range s.jobs.List() {
		if active.ID != j.ID && active.Kind == "recognition" && (active.Status == "queued" || active.Status == "running") {
			http.Error(w, "another subtitle job is active", http.StatusConflict)
			return
		}
	}

	inputDir := filepath.Join(s.dataDir, "inputs", j.ID)
	sourceURL := ""
	inputPath := ""
	if jobStringParam(j.Params, "source", "file") == "url" {
		sourceURL = j.Prompt
	} else {
		matches, _ := filepath.Glob(filepath.Join(inputDir, "source.*"))
		if len(matches) == 0 {
			http.Error(w, "saved source media is missing", http.StatusConflict)
			return
		}
		inputPath = matches[0]
	}

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
		"current_segment", "recognized_segments", "text", "cues",
	} {
		delete(workerJob.Params, key)
	}
	workerJob.Params["stage"] = "queued"
	if err := s.jobs.Save(workerJob); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	formats := jobStringSliceParam(workerJob.Params, "output_formats", s.config().Recognition.DefaultOutputFormats)
	go s.runSubtitle(
		workerJob, inputDir, inputPath, sourceURL,
		jobStringParam(workerJob.Params, "language", s.config().Recognition.DefaultLanguage),
		jobStringParam(workerJob.Params, "context", ""), formats,
		jobStringParam(workerJob.Params, "translation_mode", s.config().Recognition.DefaultTranslationMode),
		jobStringParam(workerJob.Params, "target_language", s.config().Recognition.DefaultTranslationLanguage),
		jobStringParam(workerJob.Params, "media_part", ""),
		jobStringParam(workerJob.Params, "media_source", ""),
	)
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

func (s *Server) runSubtitle(j jobs.Job, inputDir, inputPath, sourceURL, language, context string, formats []string, translationMode, targetLanguage, mediaPart, mediaSource string) {
	if s.jobCancelled(j.ID) {
		return
	}
	j.Status = "running"
	j.Params["stage"] = "media"
	j.Params["media_stage"] = "starting"
	_ = s.jobs.Save(j)
	preparedDir := filepath.Join(inputDir, "prepared")
	manifest, preparedErr := readPreparedManifest(preparedDir)
	if preparedErr == nil {
		for _, segment := range manifest.Segments {
			if info, err := os.Stat(filepath.Join(preparedDir, segment.Name)); err != nil || info.Size() == 0 {
				preparedErr = fmt.Errorf("saved media segment is missing")
				break
			}
		}
	}
	if preparedErr != nil {
		archivePath := filepath.Join(inputDir, "prepared.zip")
		fields := map[string]string{"segment_seconds": strconv.Itoa(s.config().Recognition.SegmentSeconds)}
		paths := []string{}
		if sourceURL != "" {
			fields["url"] = sourceURL
			if mediaPart != "" {
				fields["media_part"] = mediaPart
			}
			if mediaSource != "" {
				fields["media_source"] = mediaSource
			}
		} else {
			paths = []string{inputPath}
		}
		fields["request_id"] = j.ID
		endpoint := s.config().Engines["media"].Endpoint + "/v1/media/prepare"
		if err := s.prepareMediaWithProgress(&j, endpoint, fields, paths, archivePath); err != nil {
			if errors.Is(err, errJobCancelled) || s.jobCancelled(j.ID) {
				return
			}
			s.fail(j, fmt.Errorf("media preparation: %w", err))
			return
		}
		if err := extractPreparedArchive(archivePath, preparedDir); err != nil {
			s.fail(j, err)
			return
		}
		manifest, preparedErr = readPreparedManifest(preparedDir)
		if preparedErr != nil {
			s.fail(j, preparedErr)
			return
		}
	} else {
		j.Params["media_stage"] = "resuming"
		_ = s.jobs.Save(j)
	}
	if manifest.Asset != nil && manifest.Asset.ID != "" {
		j.MediaAssetID = manifest.Asset.ID
		j.MediaURL = "/api/media/assets/" + manifest.Asset.ID
		j.Params["media"] = map[string]any{
			"duration": manifest.Asset.Duration, "width": manifest.Asset.Width,
			"height": manifest.Asset.Height, "size": manifest.Asset.Size,
			"media_type": manifest.Asset.MediaType, "content_type": manifest.Asset.ContentType,
		}
		_ = s.jobs.Save(j)
	}
	j.Params["stage"] = "recognition"
	delete(j.Params, "media_stage")
	delete(j.Params, "media_percent")
	delete(j.Params, "media_downloaded_bytes")
	delete(j.Params, "media_total_bytes")
	delete(j.Params, "media_eta_seconds")
	_ = s.jobs.Save(j)
	detectedLanguage := ""
	lockedLanguage := ""
	cues := make([]subtitleCue, 0, len(manifest.Segments))
	for index := range manifest.Segments {
		if s.jobCancelled(j.ID) {
			return
		}
		segmentLanguage := language
		if isSingleLanguageAuto(language) && lockedLanguage != "" {
			segmentLanguage = lockedLanguage
		}
		text, detected, words, transcribeErr := s.transcribeSegment(filepath.Join(preparedDir, manifest.Segments[index].Name), segmentLanguage, context)
		if s.jobCancelled(j.ID) {
			return
		}
		if transcribeErr != nil {
			s.fail(j, fmt.Errorf("segment %d/%d: %w", index+1, len(manifest.Segments), transcribeErr))
			return
		}
		qualityErr := validateAlignedResult(text, words, manifest.Segments[index].Duration, isMultilingualAuto(language))
		var segmentCues []subtitleCue
		if qualityErr != nil {
			segmentCues, detected, transcribeErr = s.recoverSubtitleSegment(
				inputDir, filepath.Join(preparedDir, manifest.Segments[index].Name),
				manifest.Segments[index].Start, segmentLanguage, context,
			)
			if transcribeErr != nil {
				s.fail(j, fmt.Errorf("segment %d/%d quality check: %v; automatic split retry: %w", index+1, len(manifest.Segments), qualityErr, transcribeErr))
				return
			}
		} else {
			segmentCues = cuesFromTimestamps(text, words, manifest.Segments[index].Start)
			if len(segmentCues) == 0 && strings.TrimSpace(text) != "" {
				segmentCues = append(segmentCues, subtitleCue{Start: manifest.Segments[index].Start, End: manifest.Segments[index].End, Text: strings.TrimSpace(text)})
			}
		}
		cues = append(cues, segmentCues...)
		if isMultilingualAuto(language) {
			detectedLanguage = mergeDetectedLanguages(detectedLanguage, detected)
		} else if detectedLanguage == "" && detected != "" {
			detectedLanguage = detected
		}
		if isSingleLanguageAuto(language) && lockedLanguage == "" && detected != "" && !strings.Contains(detected, ",") {
			lockedLanguage = detected
		}
		j.Params["progress"] = index + 1
		j.Params["segments"] = len(manifest.Segments)
		_ = s.jobs.Save(j)
	}
	if len(cues) == 0 {
		s.fail(j, fmt.Errorf("recognition engine found no speech"))
		return
	}
	if translationMode != "none" {
		j.Params["stage"] = "translation"
		j.Params["translation_progress"] = 0
		j.Params["translation_total"] = (len(cues) + 7) / 8
		_ = s.jobs.Save(j)
		if err := s.translateSubtitleSegments(cues, targetLanguage, func(done, total int) {
			if s.jobCancelled(j.ID) {
				return
			}
			j.Params["translation_progress"] = done
			j.Params["translation_total"] = total
			_ = s.jobs.Save(j)
		}, func() bool { return s.jobCancelled(j.ID) }); err != nil {
			if errors.Is(err, errJobCancelled) || s.jobCancelled(j.ID) {
				return
			}
			s.fail(j, fmt.Errorf("translation: %w", err))
			return
		}
	}
	j.Params["stage"] = "finalizing"
	if s.jobCancelled(j.ID) {
		return
	}
	_ = s.jobs.Save(j)
	outputs, err := s.writeSubtitleOutputs(j.ID, cues, formats, translationMode)
	if err != nil {
		s.fail(j, err)
		return
	}
	if j.MediaAssetID != "" {
		captionName := j.ID + ".player.vtt"
		if err := os.WriteFile(s.jobs.OutputPath(captionName), []byte(renderVTT(cues, translationMode)+"\n"), 0o644); err != nil {
			s.fail(j, err)
			return
		}
		j.CaptionURL = "/api/outputs/" + captionName
	}
	preview := renderPlainText(cues, translationMode)
	if len([]rune(preview)) > 4000 {
		preview = string([]rune(preview)[:4000]) + "…"
	}
	j.Params["text"] = preview
	j.Params["segments"] = len(manifest.Segments)
	j.Params["cues"] = len(cues)
	delete(j.Params, "progress")
	delete(j.Params, "stage")
	delete(j.Params, "translation_progress")
	delete(j.Params, "translation_total")
	if detectedLanguage != "" {
		j.Params["detected_language"] = detectedLanguage
		if isSingleLanguageAuto(language) && lockedLanguage != "" {
			j.Params["locked_language"] = lockedLanguage
		}
	}
	j.Outputs = outputs
	if output, ok := outputs["txt"]; ok {
		j.OutputURL = output
	} else {
		for _, format := range formats {
			if output, ok := outputs[format]; ok {
				j.OutputURL = output
				break
			}
		}
	}
	j.Status = "completed"
	_ = s.jobs.Save(j)
}

// ResumeInterruptedJobs restarts subtitle jobs whose durable inputs or Media API
// checkpoints survived an application restart. Other generators cannot be
// resumed safely because their remote engine request state is not durable.
func (s *Server) ResumeInterruptedJobs() (resumed, failed int) {
	for _, persisted := range s.jobs.List() {
		if persisted.Status != "queued" && persisted.Status != "running" {
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
		sourceKind := jobStringParam(persisted.Params, "source", "file")
		sourceURL := ""
		inputPath := ""
		if sourceKind == "url" {
			sourceURL = persisted.Prompt
		} else {
			matches, _ := filepath.Glob(filepath.Join(inputDir, "source.*"))
			if len(matches) > 0 {
				inputPath = matches[0]
			}
			if inputPath == "" {
				persisted.Status = "failed"
				persisted.Error = "앱 재시작 후 원본 입력 파일을 찾을 수 없습니다."
				_ = s.jobs.Save(persisted)
				failed++
				continue
			}
		}

		formats := jobStringSliceParam(persisted.Params, "output_formats", s.config().Recognition.DefaultOutputFormats)
		workerJob := persisted
		workerJob.Error = ""
		workerJob.Params = make(map[string]any, len(persisted.Params))
		for key, value := range persisted.Params {
			workerJob.Params[key] = value
		}
		go s.runSubtitle(
			workerJob, inputDir, inputPath, sourceURL,
			jobStringParam(persisted.Params, "language", s.config().Recognition.DefaultLanguage),
			jobStringParam(persisted.Params, "context", ""), formats,
			jobStringParam(persisted.Params, "translation_mode", s.config().Recognition.DefaultTranslationMode),
			jobStringParam(persisted.Params, "target_language", s.config().Recognition.DefaultTranslationLanguage),
			jobStringParam(persisted.Params, "media_part", ""),
			jobStringParam(persisted.Params, "media_source", ""),
		)
		resumed++
	}
	return resumed, failed
}

func jobStringParam(params map[string]any, key, fallback string) string {
	if value, ok := params[key].(string); ok && strings.TrimSpace(value) != "" {
		return value
	}
	return fallback
}

func jobStringSliceParam(params map[string]any, key string, fallback []string) []string {
	switch values := params[key].(type) {
	case []string:
		return append([]string(nil), values...)
	case []any:
		result := make([]string, 0, len(values))
		for _, value := range values {
			if text, ok := value.(string); ok && text != "" {
				result = append(result, text)
			}
		}
		if len(result) > 0 {
			return result
		}
	}
	return append([]string(nil), fallback...)
}

func (s *Server) prepareMediaWithProgress(j *jobs.Job, endpoint string, fields map[string]string, paths []string, archivePath string) error {
	done := make(chan error, 1)
	go func() {
		done <- s.callMultipartToFileStreaming(endpoint, fields, "file", paths, archivePath)
	}()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	defer s.clearMediaProgress(j.ID)
	last := mediaProgressStatus{}
	for {
		select {
		case err := <-done:
			if s.jobCancelled(j.ID) {
				return errJobCancelled
			}
			return err
		case <-ticker.C:
			if s.jobCancelled(j.ID) {
				return errJobCancelled
			}
			progress, err := s.getMediaProgress(j.ID)
			if err != nil || progress == last {
				continue
			}
			last = progress
			j.Params["stage"] = "media"
			j.Params["media_stage"] = progress.Stage
			if progress.TotalBytes > 0 {
				j.Params["media_percent"] = progress.Percent
				j.Params["media_downloaded_bytes"] = progress.DownloadedBytes
				j.Params["media_total_bytes"] = progress.TotalBytes
			} else {
				delete(j.Params, "media_percent")
				delete(j.Params, "media_downloaded_bytes")
				delete(j.Params, "media_total_bytes")
			}
			if progress.ETASeconds > 0 {
				j.Params["media_eta_seconds"] = progress.ETASeconds
			} else {
				delete(j.Params, "media_eta_seconds")
			}
			_ = s.jobs.Save(*j)
		}
	}
}

func (s *Server) getMediaProgress(id string) (mediaProgressStatus, error) {
	endpoint := strings.TrimRight(s.config().Engines["media"].Endpoint, "/") + "/v1/media/progress/" + id
	response, err := s.health.Get(endpoint)
	if err != nil {
		return mediaProgressStatus{}, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return mediaProgressStatus{}, fmt.Errorf("media progress returned %d", response.StatusCode)
	}
	var progress mediaProgressStatus
	if err := json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&progress); err != nil {
		return mediaProgressStatus{}, err
	}
	return progress, nil
}

func (s *Server) clearMediaProgress(id string) {
	endpoint := strings.TrimRight(s.config().Engines["media"].Endpoint, "/") + "/v1/media/progress/" + id
	request, err := http.NewRequest(http.MethodDelete, endpoint, nil)
	if err != nil {
		return
	}
	response, err := s.health.Do(request)
	if err == nil {
		_ = response.Body.Close()
	}
}

func (s *Server) recoverSubtitleSegment(inputDir, sourcePath string, absoluteOffset float64, language, context string) ([]subtitleCue, string, error) {
	var lastErr error
	for _, seconds := range []int{10, 5} {
		retryDir, err := os.MkdirTemp(inputDir, fmt.Sprintf("retry-%ds-", seconds))
		if err != nil {
			return nil, "", err
		}
		archivePath := filepath.Join(retryDir, "prepared.zip")
		fields := map[string]string{"segment_seconds": strconv.Itoa(seconds)}
		endpoint := s.config().Engines["media"].Endpoint + "/v1/media/prepare"
		err = s.callMultipartToFileStreaming(endpoint, fields, "file", []string{sourcePath}, archivePath)
		if err == nil {
			err = extractPreparedArchive(archivePath, filepath.Join(retryDir, "prepared"))
		}
		var manifest preparedManifest
		if err == nil {
			manifest, err = readPreparedManifest(filepath.Join(retryDir, "prepared"))
		}
		if err != nil {
			lastErr = err
			_ = os.RemoveAll(retryDir)
			continue
		}

		var cues []subtitleCue
		detectedLanguage := ""
		for _, segment := range manifest.Segments {
			text, detected, words, transcribeErr := s.transcribeSegment(
				filepath.Join(retryDir, "prepared", segment.Name), language, context,
			)
			if transcribeErr != nil {
				err = transcribeErr
				break
			}
			if validationErr := validateAlignedResult(text, words, segment.Duration, isMultilingualAuto(language)); validationErr != nil {
				err = validationErr
				break
			}
			if isMultilingualAuto(language) {
				detectedLanguage = mergeDetectedLanguages(detectedLanguage, detected)
			} else if detectedLanguage == "" && detected != "" {
				detectedLanguage = detected
			}
			segmentCues := cuesFromTimestamps(text, words, absoluteOffset+segment.Start)
			if len(segmentCues) == 0 && strings.TrimSpace(text) != "" {
				segmentCues = append(segmentCues, subtitleCue{
					Start: absoluteOffset + segment.Start,
					End:   absoluteOffset + segment.End,
					Text:  strings.TrimSpace(text),
				})
			}
			cues = append(cues, segmentCues...)
		}
		_ = os.RemoveAll(retryDir)
		if err == nil {
			return cues, detectedLanguage, nil
		}
		lastErr = err
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("split retry produced no result")
	}
	return nil, "", lastErr
}

func (s *Server) transcribeSegment(path, language, context string) (string, string, []timedWord, error) {
	cfg := s.config()
	fields := map[string]string{"model": cfg.Recognition.Model}
	if language != "" && !isAutomaticLanguage(language) {
		fields["language"] = language
	}
	if context != "" {
		fields["prompt"] = context
	}
	data, _, err := s.callMultipart(cfg.Engines["recognition"].Endpoint+"/v1/audio/transcriptions", fields, "file", []string{path})
	if err != nil {
		return "", "", nil, err
	}
	var response struct {
		Text       string      `json:"text"`
		Language   string      `json:"language"`
		Timestamps []timedWord `json:"timestamps"`
	}
	if err := json.Unmarshal(data, &response); err != nil {
		return "", "", nil, err
	}
	return response.Text, response.Language, response.Timestamps, nil
}

func cuesFromTimestamps(transcript string, words []timedWord, offset float64) []subtitleCue {
	valid := make([]timedWord, 0, len(words))
	for _, word := range words {
		word.Text = strings.TrimSpace(word.Text)
		if word.Text != "" && word.End >= word.Start && word.Start >= 0 {
			valid = append(valid, word)
		}
	}
	if len(valid) == 0 {
		return nil
	}
	restored, exact := restoreAlignedText(strings.TrimSpace(transcript), valid)
	cues := make([]subtitleCue, 0, len(restored)/8+1)
	start := 0
	for index := range restored {
		text := cueTokenText(restored[start:index+1], exact)
		duration := restored[index].End - restored[start].Start
		if duration >= 6 || utf8.RuneCountInString(text) >= 60 || hasSentenceEnding(text) || index == len(restored)-1 {
			if text != "" {
				cues = append(cues, subtitleCue{
					Start: offset + restored[start].Start,
					End:   offset + restored[index].End,
					Text:  text,
				})
			}
			start = index + 1
		}
	}
	return cues
}

func validateAlignedResult(transcript string, words []timedWord, duration float64, allowRepeatedLyrics bool) error {
	if strings.TrimSpace(transcript) == "" || len(words) == 0 {
		return nil
	}
	outOfRange := 0
	for _, word := range words {
		if word.Start < -0.05 || word.End < word.Start || word.End > duration+0.5 {
			outOfRange++
		}
	}
	if outOfRange > 0 {
		return fmt.Errorf("aligner returned %d/%d timestamps outside %.3fs audio", outOfRange, len(words), duration)
	}
	if len(words) >= 12 {
		minimum, maximum := words[0].Start, words[0].End
		for _, word := range words[1:] {
			if word.Start < minimum {
				minimum = word.Start
			}
			if word.End > maximum {
				maximum = word.End
			}
		}
		if maximum-minimum < 0.25 {
			return fmt.Errorf("aligner collapsed %d words into %.3fs", len(words), maximum-minimum)
		}
	}
	if !allowRepeatedLyrics {
		for _, sentence := range strings.FieldsFunc(transcript, func(r rune) bool {
			return strings.ContainsRune(".!?。！？\n", r)
		}) {
			sentence = strings.TrimSpace(sentence)
			if utf8.RuneCountInString(sentence) >= 8 && strings.Count(transcript, sentence) >= 5 {
				return fmt.Errorf("ASR repeated the same sentence at least five times")
			}
		}
	}
	return nil
}

func isSingleLanguageAuto(language string) bool {
	return strings.EqualFold(strings.TrimSpace(language), "auto")
}

func isMultilingualAuto(language string) bool {
	return strings.EqualFold(strings.TrimSpace(language), autoMultilingualLanguage)
}

func isAutomaticLanguage(language string) bool {
	return isSingleLanguageAuto(language) || isMultilingualAuto(language)
}

func mergeDetectedLanguages(current, next string) string {
	seen := make(map[string]bool)
	merged := make([]string, 0, 4)
	for _, value := range []string{current, next} {
		for _, language := range strings.Split(value, ",") {
			language = strings.TrimSpace(language)
			key := strings.ToLower(language)
			if language == "" || seen[key] {
				continue
			}
			seen[key] = true
			merged = append(merged, language)
		}
	}
	return strings.Join(merged, ",")
}

// Forced Aligner는 문장부호를 제거한 어절을 반환한다. 원문에서 다음 어절까지의
// 공백과 문장부호를 앞 어절에 다시 붙여 자막 본문을 원래 표기대로 보존한다.
func restoreAlignedText(transcript string, words []timedWord) ([]timedWord, bool) {
	result := append([]timedWord(nil), words...)
	cursor := 0
	for index := range result {
		position := strings.Index(transcript[cursor:], result[index].Text)
		if position < 0 {
			return words, false
		}
		position += cursor
		if index == 0 {
			result[index].Text = transcript[:position] + result[index].Text
		} else {
			result[index-1].Text += transcript[cursor:position]
		}
		cursor = position + len(strings.TrimSpace(words[index].Text))
	}
	result[len(result)-1].Text += transcript[cursor:]
	return result, true
}

func cueTokenText(words []timedWord, exact bool) string {
	parts := make([]string, 0, len(words))
	for _, word := range words {
		parts = append(parts, word.Text)
	}
	separator := " "
	if exact {
		separator = ""
	}
	return strings.TrimSpace(strings.Join(parts, separator))
}

func hasSentenceEnding(value string) bool {
	value = strings.TrimSpace(value)
	for _, suffix := range []string{".", "?", "!", "。", "？", "！"} {
		if strings.HasSuffix(value, suffix) {
			return true
		}
	}
	return false
}

func (s *Server) translateSubtitleSegments(segments []subtitleCue, targetLanguage string, progress func(done, total int), cancelled func() bool) error {
	cfg := s.config()
	total := (len(segments) + 7) / 8
	done := 0
	for start := 0; start < len(segments); start += 8 {
		if cancelled != nil && cancelled() {
			return errJobCancelled
		}
		end := start + 8
		if end > len(segments) {
			end = len(segments)
		}
		var input strings.Builder
		for index := start; index < end; index++ {
			fmt.Fprintf(&input, "[[%04d]] %s\n", index, segments[index].Text)
		}
		systemPrompt := "You translate subtitle segments. Translate only the text into " + targetLanguage + ". Preserve every [[NNNN]] marker exactly once and in order. Do not add explanations."
		if strings.EqualFold(targetLanguage, "Korean") {
			systemPrompt = "당신은 전문 영상 자막 번역가입니다. 각 [[NNNN]] 표식을 그대로 유지하면서 뒤의 자막을 자연스러운 한국어로 번역하세요. 일본어·영어 원문을 복사하지 말고 설명 없이 번역문만 출력하세요."
		}
		payload := map[string]any{
			"model": cfg.PromptEnhancement.Model,
			"messages": []map[string]string{
				{"role": "system", "content": systemPrompt},
				{"role": "user", "content": input.String()},
			},
			"max_completion_tokens": 2048, "temperature": 0, "top_k": 1, "reasoning_effort": "none",
		}
		data, _, err := s.callJSON(cfg.Engines["prompt"].Endpoint+"/v1/chat/completions", payload)
		if cancelled != nil && cancelled() {
			return errJobCancelled
		}
		if err != nil {
			return err
		}
		var response struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(data, &response); err != nil || len(response.Choices) == 0 {
			return fmt.Errorf("translation engine returned an invalid response")
		}
		translated := parseMarkedTranslations(response.Choices[0].Message.Content)
		for index := start; index < end; index++ {
			value, ok := translated[index]
			if !ok || !validSubtitleTranslation(segments[index].Text, value, targetLanguage) {
				value, err = s.retrySubtitleTranslation(segments[index].Text, targetLanguage)
				if err != nil {
					return fmt.Errorf("segment %d: %w", index+1, err)
				}
			}
			segments[index].Translated = strings.TrimSpace(value)
		}
		done++
		if progress != nil {
			progress(done, total)
		}
	}
	return nil
}

func validSubtitleTranslation(source, translated, targetLanguage string) bool {
	source = strings.TrimSpace(source)
	translated = strings.TrimSpace(translated)
	if translated == "" {
		return false
	}
	if strings.EqualFold(source, translated) && !strings.EqualFold(targetLanguage, "Korean") {
		return false
	}
	if strings.EqualFold(targetLanguage, "Korean") && !containsHangul(source) {
		return containsHangul(translated)
	}
	return true
}

func containsHangul(value string) bool {
	for _, char := range value {
		if (char >= 0x1100 && char <= 0x11ff) || (char >= 0x3130 && char <= 0x318f) || (char >= 0xac00 && char <= 0xd7af) {
			return true
		}
	}
	return false
}

func (s *Server) retrySubtitleTranslation(source, targetLanguage string) (string, error) {
	cfg := s.config()
	systemPrompt := "You are a professional audiovisual subtitle translator. Translate the input into natural " + targetLanguage + ". Return exactly one translated subtitle and nothing else. Never copy untranslated source text."
	if strings.EqualFold(targetLanguage, "Korean") {
		systemPrompt = "당신은 전문 영상 자막 번역가입니다. 입력 자막을 자연스러운 한국어 자막 한 줄로 번역하세요. 원문을 복사하지 말고 번역문만 출력하세요."
	}
	payload := map[string]any{
		"model": cfg.PromptEnhancement.Model,
		"messages": []map[string]string{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": source},
		},
		"max_completion_tokens": 512, "temperature": 0, "top_k": 1, "reasoning_effort": "none",
	}
	data, _, err := s.callJSON(cfg.Engines["prompt"].Endpoint+"/v1/chat/completions", payload)
	if err != nil {
		return "", err
	}
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &response); err != nil || len(response.Choices) == 0 {
		return "", fmt.Errorf("translation engine returned an invalid retry response")
	}
	translated := strings.TrimSpace(response.Choices[0].Message.Content)
	if !validSubtitleTranslation(source, translated, targetLanguage) {
		return "", fmt.Errorf("translation engine did not produce %s text", targetLanguage)
	}
	return translated, nil
}

func parseMarkedTranslations(value string) map[int]string {
	result := map[int]string{}
	current := -1
	for _, line := range strings.Split(strings.TrimSpace(value), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "[[") {
			if closeIndex := strings.Index(line, "]]"); closeIndex >= 2 {
				if index, err := strconv.Atoi(line[2:closeIndex]); err == nil {
					current = index
					result[current] = strings.TrimSpace(line[closeIndex+2:])
					continue
				}
			}
		}
		if current >= 0 && line != "" {
			result[current] = strings.TrimSpace(result[current] + " " + line)
		}
	}
	return result
}

func (s *Server) writeSubtitleOutputs(id string, segments []subtitleCue, formats []string, translationMode string) (map[string]string, error) {
	outputs := map[string]string{}
	for _, format := range formats {
		var content string
		switch format {
		case "srt":
			content = renderSRT(segments, translationMode)
		case "vtt":
			content = renderVTT(segments, translationMode)
		case "timestamped_txt":
			content = renderTimestampedText(segments, translationMode)
		case "txt":
			content = renderPlainText(segments, translationMode)
		}
		name := id + "." + map[string]string{"timestamped_txt": "timestamps.txt"}[format]
		if format != "timestamped_txt" {
			name = id + "." + format
		}
		if err := os.WriteFile(s.jobs.OutputPath(name), []byte(content+"\n"), 0o644); err != nil {
			return nil, err
		}
		outputs[format] = "/api/outputs/" + name
	}
	return outputs, nil
}

func segmentText(segment subtitleCue, mode string) string {
	switch mode {
	case "translated":
		return segment.Translated
	case "bilingual":
		return segment.Text + "\n" + segment.Translated
	default:
		return segment.Text
	}
}

func renderPlainText(segments []subtitleCue, mode string) string {
	values := make([]string, 0, len(segments))
	for _, segment := range segments {
		if value := strings.TrimSpace(segmentText(segment, mode)); value != "" {
			values = append(values, value)
		}
	}
	return strings.Join(values, "\n")
}

func renderTimestampedText(segments []subtitleCue, mode string) string {
	var output strings.Builder
	for _, segment := range segments {
		fmt.Fprintf(&output, "[%s --> %s] %s\n", formatClock(segment.Start, '.'), formatClock(segment.End, '.'), segmentText(segment, mode))
	}
	return strings.TrimSpace(output.String())
}

func renderSRT(segments []subtitleCue, mode string) string {
	var output strings.Builder
	for index, segment := range segments {
		fmt.Fprintf(&output, "%d\n%s --> %s\n%s\n\n", index+1, formatClock(segment.Start, ','), formatClock(segment.End, ','), segmentText(segment, mode))
	}
	return strings.TrimSpace(output.String())
}

func renderVTT(segments []subtitleCue, mode string) string {
	var output strings.Builder
	output.WriteString("WEBVTT\n\n")
	for _, segment := range segments {
		fmt.Fprintf(&output, "%s --> %s\n%s\n\n", formatClock(segment.Start, '.'), formatClock(segment.End, '.'), segmentText(segment, mode))
	}
	return strings.TrimSpace(output.String())
}

func formatClock(seconds float64, separator rune) string {
	if seconds < 0 {
		seconds = 0
	}
	milliseconds := int64(seconds*1000 + 0.5)
	hours := milliseconds / 3600000
	milliseconds %= 3600000
	minutes := milliseconds / 60000
	milliseconds %= 60000
	secs := milliseconds / 1000
	millis := milliseconds % 1000
	return fmt.Sprintf("%02d:%02d:%02d%c%03d", hours, minutes, secs, separator, millis)
}

func (s *Server) callMultipartToFileStreaming(url string, fields map[string]string, fileField string, paths []string, output string) error {
	reader, writer := io.Pipe()
	multipartWriter := multipart.NewWriter(writer)
	producerDone := make(chan error, 1)
	go func() {
		var produceErr error
		defer func() {
			if produceErr == nil {
				produceErr = multipartWriter.Close()
			}
			_ = writer.CloseWithError(produceErr)
			producerDone <- produceErr
		}()
		keys := make([]string, 0, len(fields))
		for key := range fields {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			if produceErr = multipartWriter.WriteField(key, fields[key]); produceErr != nil {
				return
			}
		}
		for _, path := range paths {
			file, err := os.Open(path)
			if err != nil {
				produceErr = err
				return
			}
			part, err := multipartWriter.CreateFormFile(fileField, filepath.Base(path))
			if err == nil {
				_, err = io.Copy(part, file)
			}
			_ = file.Close()
			if err != nil {
				produceErr = err
				return
			}
		}
	}()
	req, err := http.NewRequest(http.MethodPost, url, reader)
	if err != nil {
		_ = reader.CloseWithError(err)
		return err
	}
	req.Header.Set("Content-Type", multipartWriter.FormDataContentType())
	resp, err := s.client.Do(req)
	if err != nil {
		_ = reader.CloseWithError(err)
		<-producerDone
		return err
	}
	defer resp.Body.Close()
	producerErr := <-producerDone
	if producerErr != nil {
		return producerErr
	}
	if resp.StatusCode/100 != 2 {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		return fmt.Errorf("engine returned %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	destination, err := os.Create(output)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(destination, resp.Body)
	closeErr := destination.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

func extractPreparedArchive(archivePath, destination string) error {
	archive, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("open prepared media: %w", err)
	}
	defer archive.Close()
	if err := os.MkdirAll(destination, 0o755); err != nil {
		return err
	}
	for _, entry := range archive.File {
		name := filepath.Clean(entry.Name)
		if filepath.Base(name) != name {
			return fmt.Errorf("invalid prepared media entry: %s", entry.Name)
		}
		source, err := entry.Open()
		if err != nil {
			return err
		}
		target, err := os.Create(filepath.Join(destination, name))
		if err == nil {
			_, err = io.Copy(target, source)
			_ = target.Close()
		}
		_ = source.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

func readPreparedManifest(directory string) (preparedManifest, error) {
	file, err := os.Open(filepath.Join(directory, "manifest.json"))
	if err != nil {
		return preparedManifest{}, err
	}
	defer file.Close()
	var manifest preparedManifest
	decoder := json.NewDecoder(bufio.NewReader(file))
	if err := decoder.Decode(&manifest); err != nil {
		return preparedManifest{}, err
	}
	if len(manifest.Segments) == 0 {
		return preparedManifest{}, fmt.Errorf("prepared media contains no audio segments")
	}
	for _, segment := range manifest.Segments {
		if filepath.Base(segment.Name) != segment.Name {
			return preparedManifest{}, fmt.Errorf("invalid segment name")
		}
	}
	if manifest.Asset != nil {
		if len(manifest.Asset.ID) != 32 {
			return preparedManifest{}, fmt.Errorf("invalid prepared media asset id")
		}
		for _, char := range manifest.Asset.ID {
			if !strings.ContainsRune("0123456789abcdef", char) {
				return preparedManifest{}, fmt.Errorf("invalid prepared media asset id")
			}
		}
	}
	return manifest, nil
}
