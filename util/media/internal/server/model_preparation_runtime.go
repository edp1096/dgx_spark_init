package server

import (
	"context"
	"encoding/json"
	"fmt"
	"mediaapp/internal/jobs"
	"net/http"
	"strings"
	"time"
)

// modelRuntimePlan is persisted in job params so preparation remains visible
// after a browser refresh and can be inspected when a job fails or is retried.
type modelRuntimePlan struct {
	Engine          string   `json:"engine"`
	Profile         string   `json:"profile"`
	Label           string   `json:"label"`
	Components      []string `json:"components"`
	Following       []string `json:"following,omitempty"`
	RuntimeOrder    []string `json:"runtime_order,omitempty"`
	RequiresSwap    bool     `json:"requires_swap,omitempty"`
	EstimateSeconds int      `json:"estimate_seconds,omitempty"`
}

func generationModelPlan(job jobs.Job) modelRuntimePlan {
	switch job.Kind {
	case "speech":
		return modelRuntimePlan{
			Engine: "speech", Profile: "qwen3-tts-custom", Label: "Qwen3 TTS 확인",
			Components:      []string{"Qwen3-TTS 1.7B CustomVoice"},
			RuntimeOrder:    []string{"상주 모델 확인", "텍스트·화자 조건 인코딩", "음성 합성", "WAV 저장"},
			EstimateSeconds: 5,
		}
	case "video":
		params := decodeVideoJobParams(job.Params)
		if params.Mode == "upscale" {
			return seedVR2RuntimePlan()
		}
		if params.Mode == "a2v" || params.Audio || len(params.AudioClips) > 0 || len(params.AudioSourceJobIDs) > 0 {
			return modelRuntimePlan{
				Engine: "video", Profile: "ltx-a2v", Label: "LTX A2V 파이프라인 준비",
				Components:   []string{"LTX-2.5 A2V dev FP8", "Gemma 4 12B", "Video/Audio VAE"},
				RuntimeOrder: []string{"파이프라인 셸 준비", "Gemma·Audio VAE 조건 인코딩", "Stage 1 DiT", "Stage 1 해제", "Stage 2 DiT", "Stage 2 해제", "Video VAE 디코딩", "VAE 해제", "MP4 저장"},
				RequiresSwap: true, EstimateSeconds: 5,
			}
		}
		components := []string{"LTX-2.5 distilled NVFP4", "Gemma 4 12B", "Video VAE"}
		if params.MotionLoRAEnabled {
			components = append(components, "LTX motion LoRA")
		}
		return modelRuntimePlan{
			Engine: "video", Profile: "ltx-distilled", Label: "LTX 영상 파이프라인 준비",
			Components:   components,
			RuntimeOrder: []string{"파이프라인 셸 준비", "Gemma 조건 인코딩", "Gemma 해제·DiT 탑재", "확산 추론", "DiT 해제·VAE 탑재", "영상·음성 디코딩", "VAE 해제", "MP4 저장"},
			RequiresSwap: true, EstimateSeconds: 5,
		}
	case "image":
		params := decodeImageJobParams(job.Params)
		switch params.Mode {
		case "upscale":
			return seedVR2RuntimePlan()
		case "garment_extract":
			return modelRuntimePlan{
				Engine: "garment", Profile: "garment-parser", Label: "의상 추출 모델 탑재",
				Components:      []string{"FASHN Human Parser"},
				RuntimeOrder:    []string{"상주 모델 확인·탑재", "의상 영역 분석", "마스크·투명 PNG 생성", "모델 상주 유지"},
				EstimateSeconds: 8,
			}
		case "face_swap":
			return modelRuntimePlan{
				Engine: "faceswap", Profile: "reactor-inswapper", Label: "ReActor 얼굴 교체 모델 탑재",
				Components:      []string{"INSWapper 128", "InsightFace buffalo_l"},
				RuntimeOrder:    []string{"얼굴 검출 모델 확인·탑재", "대상·원본 얼굴 분석", "얼굴 교체", "결과 합성·저장"},
				EstimateSeconds: 12,
			}
		case "detail_enhance":
			return modelRuntimePlan{
				Engine: "image", Profile: "krea-detail", Label: "Krea 디테일 모델 탑재",
				Components: []string{"Krea 2 checkpoint", "Qwen3VL text encoder", "Detail Enhancer LoRA", "VAE"}, EstimateSeconds: 55,
			}
		}
		return imageRuntimePlan(params, params.SequenceStrategy == "major" && params.SequencePreviousJobID != "")
	}
	return modelRuntimePlan{}
}

func seedVR2RuntimePlan() modelRuntimePlan {
	return modelRuntimePlan{
		Engine: "upscale", Profile: "seedvr2", Label: "SeedVR2 탑재",
		Components:      []string{"SeedVR2 3B FP8", "SeedVR2 VAE"},
		RuntimeOrder:    []string{"CPU 캐시 확인", "DiT·VAE GPU 이동", "업스케일 추론·디코딩", "GPU 해제", "CPU 캐시 유지", "결과 저장"},
		EstimateSeconds: 35,
	}
}

func imageRuntimePlan(params imageJobParams, major bool) modelRuntimePlan {
	checkpoint := params.Checkpoint
	if checkpoint == "" {
		checkpoint = "official-int8"
	}
	checkpointLabel := checkpoint
	estimateSeconds := 55
	if checkpoint == "official-int8" {
		checkpointLabel = "공식 INT8 ConvRot"
		estimateSeconds = 145
	} else if checkpoint == "official" {
		checkpointLabel = "공식 NVFP4 고속"
	}
	base := modelRuntimePlan{
		Engine: "image", Profile: "krea-create", Label: "Krea 생성 모델 탑재",
		Components:      []string{"Krea 2 " + checkpointLabel, "Qwen3VL FP8 text encoder", "Qwen Image VAE"},
		RuntimeOrder:    []string{"워크플로우 준비", "체크포인트·인코더·VAE·LoRA 탑재", "조건 인코딩", "확산 추론", "VAE 디코딩", "ComfyUI 캐시 유지", "결과 저장"},
		EstimateSeconds: estimateSeconds,
	}
	if major && params.SequenceDraftReady {
		return modelRuntimePlan{
			Engine: "image", Profile: "krea-identity-" + valueOr(params.IdentityModel, "convrot") + "-" + valueOr(params.IdentityEncoder, "heretic"),
			Label:        "Krea Identity Edit 탑재",
			Components:   []string{"Krea Identity Edit", valueOr(params.IdentityEncoder, "heretic") + " text encoder", "Qwen Image VAE"},
			RuntimeOrder: []string{"Identity 워크플로우 준비", "Identity DiT·인코더·VAE 탑재", "참조 조건 인코딩", "Identity 추론", "VAE 디코딩", "ComfyUI 캐시 유지"},
			RequiresSwap: true, EstimateSeconds: 55,
		}
	}
	if major {
		base.Following = []string{"Krea Identity Edit", "Heretic/기본 Identity encoder"}
		base.RequiresSwap = true
		return base
	}
	profile := "krea-create"
	label := "Krea 생성 모델 탑재"
	if params.SequenceReID {
		profile, label = "krea-reid", "Krea ReID 탑재"
		base.Components = []string{"Krea 2 INT8 ConvRot", "Qwen3VL BF16 vision encoder", "Krea ReID LoRA", "Qwen Image VAE"}
		base.RequiresSwap = true
	} else if params.Identity {
		if params.IdentityPreset == "headSwap" {
			profile, label = "krea-head-swap", "Krea 머리 전체 교체 탑재"
			base.Components = []string{"Krea 2 INT8 ConvRot", "BFS Head Swap V1.1 LoRA", "Qwen3VL FP8 text encoder", "Qwen Image VAE"}
			base.EstimateSeconds = 185
		} else {
			profile = "krea-identity-" + valueOr(params.IdentityModel, "convrot") + "-" + valueOr(params.IdentityEncoder, "heretic")
			label = "Krea Identity Edit 탑재"
			base.Components = []string{"Krea Identity Edit", valueOr(params.IdentityEncoder, "heretic") + " text encoder", "Qwen Image VAE"}
		}
	} else if params.AnyPaint {
		profile, label = "krea-anypaint", "Krea AnyPaint 탑재"
		base.Components = append(base.Components, "AnyPaint LoRA")
	} else if params.NK2E {
		profile, label = "krea-nk2e-"+valueOr(params.NK2EMode, "edit"), "Krea 구조 제어 모델 탑재"
		base.Components = append(base.Components, "NK2E "+valueOr(params.NK2EMode, "edit")+" LoRA")
	} else if params.StyleReference {
		profile, label = "krea-style-reference", "Krea 스타일 참조 모델 탑재"
		base.Components = append(base.Components, "Style Reference LoRA")
	} else if params.Depth {
		profile, label = "krea-depth", "Krea Depth 모델 탑재"
		base.Components = append(base.Components, "Depth Control LoRA")
	} else if params.Vision {
		profile, label = "krea-vision-"+valueOr(params.VisionMode, "descriptor"), "Krea 비전 참조 모델 탑재"
		base.Components[1] = "Qwen3VL BF16 vision encoder"
	}
	base.Profile, base.Label = profile, label
	return base
}

func modelPlanMap(plan modelRuntimePlan) map[string]any {
	data, _ := json.Marshal(plan)
	result := map[string]any{}
	_ = json.Unmarshal(data, &result)
	result["status"] = "pending"
	return result
}

func setInitialModelPlan(job *jobs.Job) {
	ensureJobParams(job)
	plan := generationModelPlan(*job)
	if plan.Engine != "" {
		job.Params["model_plan"] = modelPlanMap(plan)
	}
}

func (s *Server) beginModelPreparation(job *jobs.Job, plan modelRuntimePlan) error {
	ensureJobParams(job)
	now := time.Now()
	job.Params["stage"] = "model-preparing"
	job.Params["stage_started_at"] = now.Format(time.RFC3339Nano)
	job.Params["model_prepare_started_at"] = now.Format(time.RFC3339Nano)
	job.Params["model_prepare_profile"] = plan.Profile
	job.Params["model_prepare_label"] = plan.Label
	job.Params["model_prepare_estimate_seconds"] = plan.EstimateSeconds
	modelPlan := modelPlanMap(plan)
	modelPlan["status"] = "preparing"
	job.Params["model_plan"] = modelPlan
	return s.jobs.Save(*job)
}

func (s *Server) completeModelPreparation(job *jobs.Job, plan modelRuntimePlan, result map[string]any) error {
	ensureJobParams(job)
	if current, ok := s.jobs.Get(job.ID); ok {
		mergeObservedRuntime(&job.Params, current.Params)
	}
	now := time.Now()
	started, _ := time.Parse(time.RFC3339Nano, stringParam(job.Params, "model_prepare_started_at", ""))
	seconds := now.Sub(started).Seconds()
	if started.IsZero() || seconds < 0 {
		seconds = 0
	}
	if reported, ok := numberValue(result["load_seconds"]); ok && reported >= 0 {
		seconds = reported
	}
	job.Params["model_prepare_completed_at"] = now.Format(time.RFC3339Nano)
	job.Params["model_prepare_seconds"] = seconds
	job.Params["generation_started_at"] = now.Format(time.RFC3339Nano)
	job.Params["stage_started_at"] = now.Format(time.RFC3339Nano)
	job.Params["stage"] = "running"
	modelPlan := modelPlanMap(plan)
	modelPlan["status"] = "ready"
	modelPlan["load_seconds"] = seconds
	if warm, ok := result["warm"].(bool); ok {
		modelPlan["warm"] = warm
	}
	for _, key := range []string{"resident", "preparation_scope", "phase_swapped", "note"} {
		if value, ok := result[key]; ok {
			modelPlan[key] = value
		}
	}
	job.Params["model_plan"] = modelPlan
	return s.jobs.Save(*job)
}

func numberValue(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case json.Number:
		result, err := typed.Float64()
		return result, err == nil
	default:
		return 0, false
	}
}

func (s *Server) callRuntimePrepare(ctx context.Context, endpoint string, payload map[string]any) (map[string]any, error) {
	data, _, err := s.callJSONContext(ctx, strings.TrimRight(endpoint, "/")+"/v1/models/runtime/prepare", payload)
	if err != nil {
		return nil, err
	}
	result := map[string]any{}
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("invalid model preparation response: %w", err)
	}
	return result, nil
}

func (s *Server) runtimePrepareSupported(ctx context.Context, endpoint string) (bool, error) {
	_ = ctx
	endpoint = strings.TrimRight(endpoint, "/")
	s.runtimeCapabilityMu.RLock()
	supported := s.runtimeCapabilities[endpoint]
	s.runtimeCapabilityMu.RUnlock()
	return supported, nil
}

func (s *Server) prepareSimpleRuntime(ctx context.Context, job *jobs.Job, plan modelRuntimePlan) error {
	if err := s.beginModelPreparation(job, plan); err != nil {
		return err
	}
	endpoint := s.config().Engines[plan.Engine].Endpoint
	supported, err := s.runtimePrepareSupported(ctx, endpoint)
	if err != nil {
		return err
	}
	if !supported {
		return s.completeModelPreparation(job, plan, map[string]any{"load_seconds": 0.0, "warm": true, "legacy": true})
	}
	observer := s.startRuntimeObserver(ctx, job.ID, endpoint)
	result, err := s.callRuntimePrepare(ctx, endpoint, map[string]any{"operation_id": job.ID})
	observer.Stop()
	if err != nil {
		return err
	}
	return s.completeModelPreparation(job, plan, result)
}

func (s *Server) prepareVideoRuntime(ctx context.Context, job *jobs.Job, params videoJobParams) error {
	plan := generationModelPlan(*job)
	if err := s.beginModelPreparation(job, plan); err != nil {
		return err
	}
	endpoint := s.config().Engines["video"].Endpoint
	supported, err := s.runtimePrepareSupported(ctx, endpoint)
	if err != nil {
		return err
	}
	if !supported {
		return s.completeModelPreparation(job, plan, map[string]any{"load_seconds": 0.0, "warm": true, "legacy": true})
	}
	pipeline := "distilled"
	if plan.Profile == "ltx-a2v" {
		pipeline = "a2v"
	}
	motionStrength := 0.0
	if params.MotionLoRAEnabled {
		motionStrength = params.MotionLoRAStrength
	}
	observer := s.startRuntimeObserver(ctx, job.ID, endpoint)
	result, err := s.callRuntimePrepare(ctx, endpoint, map[string]any{
		"pipeline": pipeline, "motion_lora_strength": motionStrength, "operation_id": job.ID,
	})
	observer.Stop()
	if err != nil {
		return err
	}
	return s.completeModelPreparation(job, plan, result)
}

func (s *Server) prepareKreaRuntime(ctx context.Context, job *jobs.Job, execution generationImageExecution, plan modelRuntimePlan) error {
	if err := s.beginModelPreparation(job, plan); err != nil {
		return err
	}
	backend := s.config().Image.Backends[execution.mode]
	supported, err := s.runtimePrepareSupported(ctx, backend.Endpoint)
	if err != nil {
		return err
	}
	if !supported {
		return s.completeModelPreparation(job, plan, map[string]any{"load_seconds": 0.0, "warm": true, "legacy": true})
	}
	started := time.Now()
	observer := s.startRuntimeObserver(ctx, job.ID, backend.Endpoint)
	result, err := s.prepareKreaCreate(ctx, backend, job.ID, execution.prompt, execution.width, execution.height, execution.seed, execution.options, plan.Profile)
	observer.Stop()
	if err != nil {
		return err
	}
	if _, ok := result["load_seconds"]; !ok {
		result["load_seconds"] = time.Since(started).Seconds()
	}
	return s.completeModelPreparation(job, plan, result)
}

func (s *Server) prepareKreaRequest(ctx context.Context, job *jobs.Job, plan modelRuntimePlan, request map[string]any) error {
	if err := s.beginModelPreparation(job, plan); err != nil {
		return err
	}
	endpoint := s.config().Image.Backends["create"].Endpoint
	supported, err := s.runtimePrepareSupported(ctx, endpoint)
	if err != nil {
		return err
	}
	if !supported {
		return s.completeModelPreparation(job, plan, map[string]any{"load_seconds": 0.0, "warm": true, "legacy": true})
	}
	prepareRequest := make(map[string]any, len(request)+2)
	for key, value := range request {
		prepareRequest[key] = value
	}
	prepareRequest["steps"] = 1
	prepareRequest["prepare_only"] = true
	prepareRequest["runtime_profile"] = plan.Profile
	prepareRequest["operation_id"] = job.ID
	started := time.Now()
	observer := s.startRuntimeObserver(ctx, job.ID, endpoint)
	data, _, err := s.callJSONContext(ctx, endpoint+"/v1/images/generations", prepareRequest)
	observer.Stop()
	if err != nil {
		return err
	}
	result := map[string]any{}
	_ = json.Unmarshal(data, &result)
	if _, ok := result["load_seconds"]; !ok {
		result["load_seconds"] = time.Since(started).Seconds()
	}
	return s.completeModelPreparation(job, plan, result)
}

func (s *Server) markResidentRuntimeReady(job *jobs.Job, plan modelRuntimePlan) error {
	if err := s.beginModelPreparation(job, plan); err != nil {
		return err
	}
	return s.completeModelPreparation(job, plan, map[string]any{"load_seconds": 0.0, "warm": true})
}

func (s *Server) prepareRecognitionRuntime(ctx context.Context, job *jobs.Job) error {
	plan := modelRuntimePlan{
		Engine: "recognition", Profile: "qwen3-asr-aligner", Label: "음성 인식 모델 확인",
		Components:      []string{"Qwen3-ASR 1.7B", "Qwen3 Forced Aligner 0.6B"},
		RuntimeOrder:    []string{"상주 모델 확인", "구간별 음성 인식", "강제 정렬", "번역", "자막·스크립트 저장"},
		EstimateSeconds: 5,
	}
	if err := s.beginModelPreparation(job, plan); err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(s.config().Engines["recognition"].Endpoint, "/")+"/health", nil)
	if err != nil {
		return err
	}
	response, err := s.client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		return s.completeModelPreparation(job, plan, map[string]any{"load_seconds": 0.0, "warm": true, "legacy": true})
	}
	return s.completeModelPreparation(job, plan, map[string]any{"load_seconds": 0.0, "warm": true})
}
