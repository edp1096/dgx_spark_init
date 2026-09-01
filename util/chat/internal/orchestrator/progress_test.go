package orchestrator

import (
	"math"
	"testing"
)

func TestInferProgress(t *testing.T) {
	component := Component{ID: "flash-next", ProgressKind: "vllm"}
	info := inferProgress(component, "Loading safetensors checkpoint shards:  10% Completed | 20/206 [00:03<01:10,  2.50it/s]\nLoading safetensors checkpoint shards:  57% Completed | 118/206 [00:24<00:20,  4.33it/s]")
	if info.Phase != "Flash Next 체크포인트 적재" || info.Detail != "118/206 샤드 · SSD에서 통합메모리로 읽는 중" || math.Abs(info.Progress-.4906) > .00001 || info.ETA != "00:20" {
		t.Fatalf("unexpected progress: %#v", info)
	}
	info = inferProgress(component, "Loading weights took 79.42 GiB memory\nGPU KV cache size: 193,218 tokens")
	if info.Phase != "KV 캐시 할당" || info.Progress != .89 {
		t.Fatalf("unexpected warmup: %#v", info)
	}
}

func TestInferGemmaDFlashProgress(t *testing.T) {
	component := Component{ID: "gemma31", ProgressKind: "sglang"}
	logs := "Load weight begin.\nMulti-thread loading shards: 100% Completed | 1/1 [00:24<00:00]\nLoad weight end. type=Gemma4ForConditionalGeneration\nLoad weight begin.\nMulti-thread loading shards: 50% Completed | 1/2 [00:08<00:08]"
	info := inferProgress(component, logs)
	if info.Key != "draft-weights" || info.Phase != "DFlash 가중치 적재" || info.Progress != .61 {
		t.Fatalf("unexpected DFlash progress: %#v", info)
	}
	info = inferProgress(component, "Prefill batch, #new-seq: 1, #new-token: 1, cuda graph: False")
	if info.Key != "draft-warmup" || info.Progress != .97 {
		t.Fatalf("unexpected DFlash warmup: %#v", info)
	}
}

func TestInferQwenDFlash2AndGraphProgress(t *testing.T) {
	component := Component{ID: "qwen27", ProgressKind: "sglang"}
	logs := "Capture target verify CUDA graph begin\nMAX_FUSED_QKV_SPLIT_DIM: <torch._dynamo warning potential risk>\nCapturing batches (bs=1):  50%|xxxxx| 1/2 [00:01<00:01, 1.00s/it]"
	info := inferProgress(component, logs)
	if info.Key != "cuda-graph-target" || info.Phase != "Target CUDA Graph 캡처" || info.Progress != .89 || info.ETA != "00:01" {
		t.Fatalf("unexpected Qwen graph progress: %#v", info)
	}
}

func TestInferQwenTorchCompileBeforeGraphCapture(t *testing.T) {
	component := Component{ID: "qwen27", ProgressKind: "sglang"}
	logs := "Capture target verify CUDA graph begin\nCapturing batches: 0%| | 0/2 [00:00<?, ?it/s]\ntorch._dynamo.utils.warn_once(msg)"
	info := inferProgress(component, logs)
	if info.Key != "torch-compile-target" || info.Phase != "Target CUDA Graph 커널 컴파일" || info.ETA != "" {
		t.Fatalf("unexpected Qwen compile progress: %#v", info)
	}
}

func TestInferDraftGraphUsesOnlyDraftSection(t *testing.T) {
	component := Component{ID: "gemma31", ProgressKind: "sglang"}
	logs := "Capture target verify CUDA graph begin\nCapturing batches: 100%|xxxxx| 2/2 [00:10<00:00, 1.00s/it]\nCapture target verify CUDA graph end\nCapture draft verify CUDA graph begin\nCapturing batches: 0%| | 0/2 [00:00<?, ?it/s]"
	info := inferProgress(component, logs)
	if info.Key != "cuda-graph-draft" || info.Phase != "DFlash CUDA Graph 캡처" || info.Progress != .84 || info.ETA != "" {
		t.Fatalf("unexpected draft graph progress: %#v", info)
	}
}

func TestETARejectsTorchAngleBracketWarnings(t *testing.T) {
	logs := "Capturing batches: 0%| | 0/2 [00:00<?, ?it/s]\nMAX_FUSED_QKV_SPLIT_DIM: <torch._dynamo warning potential risk>"
	if eta := etaForCurrentProgressLine(logs, barProgressPattern); eta != "" {
		t.Fatalf("warning text must not become an ETA: %q", eta)
	}
}

func TestImageWrapperStartupIsNotModelReady(t *testing.T) {
	component := Component{ID: "flux2", ProgressKind: "comfy"}
	info := inferProgress(component, "Application startup complete.\nUvicorn running on http://0.0.0.0:8691")
	if info.Key != "container" || info.Progress != .12 {
		t.Fatalf("wrapper startup must not report the image model ready: %#v", info)
	}
}

func TestStartupFailureDetectsCUDAOOM(t *testing.T) {
	logs := "torch.AcceleratorError: CUDA error: out of memory"
	if failure := startupFailure(logs); failure == "" {
		t.Fatal("CUDA OOM must stop startup polling")
	}
}
