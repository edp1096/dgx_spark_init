# analyze_model.py

FP8 모델의 양자화 포맷을 분석하는 도구.

## 요구사항

```bash
pip install safetensors huggingface_hub
```

- `index` 모드: 추가 패키지 없이 동작 (순수 JSON 파싱)
- `safetensors` 모드: `safetensors`, `torch` 필요
- `hf` 모드: `huggingface_hub` 필요

## 사용법

### 1. index — JSON 파일 분석

```bash
python analyze_model.py index /path/to/model.safetensors.index.json
```

로컬 PC에서 실행 가능. 텐서 이름 패턴만으로 아래 항목을 파악한다:

- Scale suffix 종류 및 개수 (`_scale_inv`, `_scale`, `.weight_scale` 등)
- MoE expert 수, 레이어 범위, 네이밍 패턴
- Top-level prefix (`language_model.`, `model.` 등)
- Layer 0의 비expert/비scale 텐서 구조
- Weight↔Scale 페어링 체크 및 unpaired 목록

### 2. safetensors — 실제 파일 분석

```bash
python analyze_model.py safetensors /path/to/model-00001-of-00005.safetensors
```

safetensors 파일을 열어서 실제 텐서 데이터를 분석한다. **샤드 하나만 있어도 동작**한다.

- dtype 분포 (float8_e4m3fn, bfloat16, float32 등)
- shape별 텐서 그룹핑
- Scale 텐서의 값 범위(min/max/mean)
- Block size 추정
- 양자화 유형 판별 (per-tensor, per-channel, block-wise)

### 3. hf — HuggingFace 모델명으로 분석

```bash
python analyze_model.py hf meta-llama/Llama-3.1-8B-Instruct-FP8
```

`model.safetensors.index.json`을 자동 다운로드한 후 `index` 분석을 수행한다. gated 모델은 `huggingface-cli login` 필요.

## 출력 예시

```
Total tensors: 92425
Shards: 163

=== Scale / Meta tensors ===
  _scale_inv: 45932 tensors

=== MoE Experts ===
  Expert count: 256 (id: 0~255)
  MoE layers: 3~61 (59 layers)
  Sample (layer 3, expert 0):
    model.layers.3.mlp.experts.0.down_proj.weight
    model.layers.3.mlp.experts.0.down_proj.weight_scale_inv
    model.layers.3.mlp.experts.0.gate_proj.weight
    model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv

=== Top-level prefix ===
  model: 92424 tensors
  lm_head: 1 tensors

=== Layer 0 structure (non-expert, non-scale) ===
  model.layers.0.input_layernorm.weight
  model.layers.0.mlp.down_proj.weight
  ...

=== Scale pairing check ===
  Paired (weight+scale): 45932
  Unpaired (no scale, e.g. embed/norm): 561
```

## quantize_fp8.py와의 관계

이 도구로 먼저 FP8 모델의 포맷을 확인한 후, `quantize_fp8.py`의 호환성을 판단한다.

확인 포인트:
1. **Scale suffix** — `SCALE_SUFFIXES` 목록에 있는지
2. **MoE expert 패턴** — `MOE_EXPERT_PATTERN` regex에 매칭되는지
3. **Block size** — `safetensors` 모드에서 추정된 값이 128인지
