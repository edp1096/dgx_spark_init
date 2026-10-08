package server

import (
	"fmt"
	"math"
	"mediaapp/internal/jobs"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func (s *Server) createVideo(w http.ResponseWriter, r *http.Request) {
	cfg := s.config()
	if err := r.ParseMultipartForm(320 << 20); err != nil {
		http.Error(w, "invalid or oversized form", http.StatusBadRequest)
		return
	}
	effectivePrompt := strings.TrimSpace(r.FormValue("prompt"))
	if effectivePrompt == "" {
		http.Error(w, "prompt is required", http.StatusBadRequest)
		return
	}
	originalPrompt := strings.TrimSpace(r.FormValue("original_prompt"))
	if originalPrompt == "" {
		originalPrompt = effectivePrompt
	}
	width := formInt(r, "width", cfg.Video.DefaultWidth)
	height := formInt(r, "height", cfg.Video.DefaultHeight)
	frames := formInt(r, "num_frames", cfg.Video.DefaultFrames)
	fps := formFloat64(r, "fps", cfg.Video.DefaultFPS)
	seed := formInt64(r, "seed", -1)
	strength := formFloat64(r, "image_strength", 1)
	if width < 256 || height < 256 || width > 1920 || height > 1920 || width%64 != 0 || height%64 != 0 {
		http.Error(w, "width and height must be between 256 and 1920 and divisible by 64", http.StatusBadRequest)
		return
	}
	if frames < 9 || (frames-1)%8 != 0 {
		http.Error(w, "num_frames must be 8*k+1 and at least 9", http.StatusBadRequest)
		return
	}
	if fps < 1 || fps > 60 {
		http.Error(w, "fps must be between 1 and 60", http.StatusBadRequest)
		return
	}
	id := newID()
	inputDir := filepath.Join(s.dataDir, "inputs", id)
	loadImage := func(uploadField, reuseField, dir string) (string, error) {
		paths, err := saveUploads(r, uploadField, dir, 1)
		if err != nil {
			return "", err
		}
		paths, err = s.appendReusedImageInputs(r, reuseField, dir, 1, paths)
		if err != nil {
			return "", err
		}
		if len(paths) == 0 {
			return "", nil
		}
		return paths[0], nil
	}

	startPath, err := loadImage("start_image", "reuse_start_image", filepath.Join(inputDir, "start"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// Keep the original single-image API compatible with older clients.
	if startPath == "" {
		legacy, legacyErr := saveUploads(r, "image", filepath.Join(inputDir, "start"), 1)
		if legacyErr != nil {
			http.Error(w, legacyErr.Error(), http.StatusBadRequest)
			return
		}
		if len(legacy) > 0 {
			startPath = legacy[0]
		}
	}
	endPath, err := loadImage("end_image", "reuse_end_image", filepath.Join(inputDir, "end"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	audioCount := formInt(r, "audio_count", 0)
	legacyAudioSourceJobID := strings.TrimSpace(r.FormValue("reuse_audio_job"))
	if audioCount == 0 && legacyAudioSourceJobID != "" {
		audioCount = 1
	}
	if audioCount < 0 || audioCount > 8 {
		http.Error(w, "audio_count must be between 0 and 8", http.StatusBadRequest)
		return
	}
	audioPaths := make([]string, 0, audioCount)
	audioClips := make([]savedVideoAudioClip, 0, audioCount)
	audioSourceJobIDs := make([]string, 0, audioCount)
	videoDuration := float64(frames) / fps
	for i := 0; i < audioCount; i++ {
		audioSourceJobID := strings.TrimSpace(r.FormValue(fmt.Sprintf("reuse_audio_job_%d", i)))
		if i == 0 && audioSourceJobID == "" {
			audioSourceJobID = legacyAudioSourceJobID
		}
		if audioSourceJobID == "" {
			http.Error(w, fmt.Sprintf("audio clip %d source is required", i+1), http.StatusBadRequest)
			return
		}
		source, ok := s.jobs.Get(audioSourceJobID)
		if !ok || source.Kind != "speech" || source.Status != "completed" || source.OutputURL == "" {
			http.Error(w, "selected A2V audio is no longer available", http.StatusBadRequest)
			return
		}
		start := formFloat64(r, fmt.Sprintf("audio_start_%d", i), 0)
		duration := formFloat64(r, fmt.Sprintf("audio_duration_%d", i), 0)
		if math.IsNaN(start) || math.IsInf(start, 0) || start < 0 || start >= videoDuration {
			http.Error(w, fmt.Sprintf("audio clip %d start must be within the video", i+1), http.StatusBadRequest)
			return
		}
		if math.IsNaN(duration) || math.IsInf(duration, 0) || duration < 0 {
			http.Error(w, fmt.Sprintf("audio clip %d duration is invalid", i+1), http.StatusBadRequest)
			return
		}
		sourcePath := s.jobs.OutputPath(filepath.Base(source.OutputURL))
		audioDir := filepath.Join(inputDir, fmt.Sprintf("audio-%d", i))
		if err := os.MkdirAll(audioDir, 0o755); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		audioPath := filepath.Join(audioDir, filepath.Base(sourcePath))
		if err := linkOrCopyFile(sourcePath, audioPath); err != nil {
			http.Error(w, "could not preserve selected A2V audio: "+err.Error(), http.StatusBadRequest)
			return
		}
		audioPaths = append(audioPaths, audioPath)
		audioSourceJobIDs = append(audioSourceJobIDs, audioSourceJobID)
		audioClips = append(audioClips, savedVideoAudioClip{Index: i, SourceJobID: audioSourceJobID, Start: start, Duration: duration})
	}

	conditions := make([]videoConditioningInput, 0, 10)
	usedFrames := map[int]bool{}
	if startPath != "" {
		conditions = append(conditions, videoConditioningInput{Path: startPath, FrameIdx: 0, Strength: strength, Role: "start"})
		usedFrames[0] = true
	}
	keyframeCount := formInt(r, "keyframe_count", 0)
	if keyframeCount < 0 || keyframeCount > 8 {
		http.Error(w, "keyframe_count must be between 0 and 8", http.StatusBadRequest)
		return
	}
	for i := 0; i < keyframeCount; i++ {
		keyframePath, loadErr := loadImage(
			fmt.Sprintf("keyframe_image_%d", i),
			fmt.Sprintf("reuse_keyframe_image_%d", i),
			filepath.Join(inputDir, fmt.Sprintf("keyframe-%d", i)),
		)
		if loadErr != nil {
			http.Error(w, loadErr.Error(), http.StatusBadRequest)
			return
		}
		if keyframePath == "" {
			http.Error(w, fmt.Sprintf("keyframe %d image is required", i+1), http.StatusBadRequest)
			return
		}
		seconds := formFloat64(r, fmt.Sprintf("keyframe_time_%d", i), -1)
		frameIdx := int(seconds*fps + 0.5)
		if seconds < 0 || frameIdx <= 0 || frameIdx >= frames-1 {
			http.Error(w, fmt.Sprintf("keyframe %d time must be between the start and end frames", i+1), http.StatusBadRequest)
			return
		}
		if usedFrames[frameIdx] {
			http.Error(w, fmt.Sprintf("keyframe %d overlaps another conditioning frame", i+1), http.StatusBadRequest)
			return
		}
		keyframeStrength := formFloat64(r, fmt.Sprintf("keyframe_strength_%d", i), 1)
		if keyframeStrength < 0 || keyframeStrength > 1 {
			http.Error(w, fmt.Sprintf("keyframe %d strength must be between 0 and 1", i+1), http.StatusBadRequest)
			return
		}
		conditions = append(conditions, videoConditioningInput{Path: keyframePath, FrameIdx: frameIdx, Strength: keyframeStrength, Role: "keyframe"})
		usedFrames[frameIdx] = true
	}
	if endPath != "" {
		conditions = append(conditions, videoConditioningInput{Path: endPath, FrameIdx: frames - 1, Strength: formFloat64(r, "end_image_strength", 1), Role: "end"})
	}
	for _, condition := range conditions {
		if condition.Strength < 0 || condition.Strength > 1 {
			http.Error(w, condition.Role+" image strength must be between 0 and 1", http.StatusBadRequest)
			return
		}
	}
	savedConditions := make([]savedVideoCondition, 0, len(conditions))
	keyframeIndex := 0
	for _, condition := range conditions {
		saved := savedVideoCondition{Role: condition.Role, FrameIdx: condition.FrameIdx, Strength: condition.Strength}
		if condition.Role == "keyframe" {
			saved.Index = keyframeIndex
			keyframeIndex++
		}
		savedConditions = append(savedConditions, saved)
	}
	videoMode := "generate"
	if len(audioPaths) > 0 {
		videoMode = "a2v"
	}
	firstAudioSourceJobID := ""
	if len(audioSourceJobIDs) > 0 {
		firstAudioSourceJobID = audioSourceJobIDs[0]
	}
	params := newVideoJobParams()
	params.Mode = videoMode
	params.Width, params.Height = width, height
	params.Frames, params.FPS, params.Seed = frames, fps, seed
	params.ImageStrength = strength
	params.Image = len(conditions) > 0
	params.StartImage, params.EndImage = startPath != "", endPath != ""
	params.Keyframes, params.Conditions = keyframeCount, savedConditions
	params.Audio = len(audioPaths) > 0
	params.AudioSourceJobID, params.AudioSourceJobIDs = firstAudioSourceJobID, audioSourceJobIDs
	params.AudioClips = audioClips
	params.MotionLoRAEnabled = cfg.Video.DefaultMotionLoRAEnabled
	params.MotionLoRAStrength = cfg.Video.DefaultMotionLoRAStrength
	params.AccelerationRequested = cfg.Video.Acceleration
	params.EnhancedPrompt = valueIfDifferent(effectivePrompt, originalPrompt)
	params.Stage = "queued"
	now := time.Now()
	params.QueuedAt = now.Format(time.RFC3339Nano)
	j := jobs.Job{
		ID: id, Kind: "video", Status: "queued", Prompt: originalPrompt,
		Params: params.toMap(), CreatedAt: now,
	}
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.wakeGenerationQueue()
	writeJSON(w, http.StatusAccepted, j)
}
