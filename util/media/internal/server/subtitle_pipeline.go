package server

import (
	stdcontext "context"
	"errors"
	"fmt"
	"mediaapp/internal/jobs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

func (s *Server) runSubtitle(j jobs.Job, inputDir, inputPath, sourceURL, language, context string, formats []string, translationMode, targetLanguage, mediaPart, mediaSource string) {
	if s.jobCancelled(j.ID) {
		return
	}
	ensureJobParams(&j)
	j.Params["stage"] = "media"
	j.Params["media_stage"] = "starting"
	j.Params["stage_started_at"] = time.Now().Format(time.RFC3339Nano)
	_ = s.saveJobPreservingRuntime(j)
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
			// Graceful shutdown requeues the durable job before cancelling its
			// remote media request. Do not let the retiring worker overwrite that
			// recovery state with the expected cancellation response.
			if persisted, ok := s.jobs.Get(j.ID); ok && persisted.Status == "queued" {
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
		_ = s.saveJobPreservingRuntime(j)
	}
	if manifest.Asset != nil && manifest.Asset.ID != "" {
		j.MediaAssetID = manifest.Asset.ID
		j.MediaURL = "/api/media/assets/" + manifest.Asset.ID
		j.Params["media"] = map[string]any{
			"duration": manifest.Asset.Duration, "width": manifest.Asset.Width,
			"height": manifest.Asset.Height, "size": manifest.Asset.Size,
			"media_type": manifest.Asset.MediaType, "content_type": manifest.Asset.ContentType,
		}
		_ = s.saveJobPreservingRuntime(j)
	}
	if err := s.prepareRecognitionRuntime(stdcontext.Background(), &j); err != nil {
		s.fail(j, fmt.Errorf("recognition model preparation: %w", err))
		return
	}
	j.Params["stage"] = "recognition"
	j.Params["stage_started_at"] = time.Now().Format(time.RFC3339Nano)
	delete(j.Params, "media_stage")
	delete(j.Params, "media_percent")
	delete(j.Params, "media_downloaded_bytes")
	delete(j.Params, "media_total_bytes")
	delete(j.Params, "media_eta_seconds")
	_ = s.saveJobPreservingRuntime(j)
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
		s.publishLocalRuntimePhase(
			j.ID, "sampling", "Qwen3-ASR·Forced Aligner",
			fmt.Sprintf("음성 인식·정렬 %d/%d 구간", index+1, len(manifest.Segments)),
			float64(index)/float64(max(1, len(manifest.Segments))), "retain", runtimeBoolPointer(true),
		)
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
		_ = s.saveJobPreservingRuntime(j)
	}
	if len(cues) == 0 {
		s.fail(j, fmt.Errorf("recognition engine found no speech"))
		return
	}
	s.publishLocalRuntimePhase(
		j.ID, "finalizing", "자막 출력", "인식 결과를 자막·스크립트로 정리", .96, "retain", runtimeBoolPointer(true),
	)
	if translationMode != "none" {
		j.Params["stage"] = "translation"
		j.Params["stage_started_at"] = time.Now().Format(time.RFC3339Nano)
		j.Params["translation_progress"] = 0
		j.Params["translation_total"] = (len(cues) + 7) / 8
		delete(j.Params, "translation_warnings")
		delete(j.Params, "translation_warning_count")
		_ = s.saveJobPreservingRuntime(j)
		warnings, err := s.translateSubtitleSegments(cues, targetLanguage, func(done, total int) {
			if s.jobCancelled(j.ID) {
				return
			}
			j.Params["translation_progress"] = done
			j.Params["translation_total"] = total
			_ = s.saveJobPreservingRuntime(j)
		}, func() bool { return s.jobCancelled(j.ID) })
		if err != nil {
			if errors.Is(err, errJobCancelled) || s.jobCancelled(j.ID) {
				return
			}
			s.fail(j, fmt.Errorf("translation: %w", err))
			return
		}
		if len(warnings) > 0 {
			j.Params["translation_warnings"] = warnings
			j.Params["translation_warning_count"] = len(warnings)
			_ = s.saveJobPreservingRuntime(j)
		}
	}
	j.Params["stage"] = "finalizing"
	j.Params["stage_started_at"] = time.Now().Format(time.RFC3339Nano)
	if s.jobCancelled(j.ID) {
		return
	}
	_ = s.saveJobPreservingRuntime(j)
	if err := s.writeSubtitleCueArchive(j.ID, cues); err != nil {
		s.fail(j, err)
		return
	}
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
	delete(j.Params, "stage_started_at")
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
	s.publishLocalRuntimePhase(
		j.ID, "completed", "Qwen3-ASR·자막 출력", "받아쓰기·정렬·자막 저장 완료", 1, "retain", runtimeBoolPointer(true),
	)
	transitionJobCompleted(&j, j.OutputURL)
	_ = s.saveJobPreservingRuntime(j)
}
