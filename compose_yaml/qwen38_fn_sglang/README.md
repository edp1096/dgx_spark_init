# Qwen3.8 Flash-Next · SGLang TP1

1 Spark에서 local-inference-lab NVFP4/MXFP8 모델 실행.
이미지는 `dgx-sglang-qwen38-qad:sm121-v4`이며, 양자화·PLE 지원과 라우터 오류 수정이 포함돼 있다.

## 준비 및 빌드

**Dockerfile만 받지 말고 이 폴더 전체를 준비.** 저장소 루트에서 실행:

```sh
cd compose_yaml/qwen38_fn_sglang
hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \
  --revision 7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd \
  --cache-dir "$HOME/.cache/huggingface/hub"
docker compose build
```

Compose가 `qad-tp1` 빌드 대상을 선택한다. 직접 `docker build`를 쓰면
`--target qad-tp1`을 지정해야 한다. 가중치는 이미지에 포함되지 않는다.

## 모델 선택

| 선택 파일 | 모델 |
| --- | --- |
| `model.official.env` | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (기본값) |
| `model.abliterated.env` | `huginnfork/Qwen3.8-Flash-Next-NVFP4-Abliterated` |
| `model.huihui-lil.env` | `edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD` |

Huginnfork 모델을 선택하려면 먼저 다운로드한다.

```sh
hf download huginnfork/Qwen3.8-Flash-Next-NVFP4-Abliterated \
  --revision 93a1b466ce773185f21a49d1649b7933ce0fc910 \
  --cache-dir "$HOME/.cache/huggingface/hub"
```

Huihui/LIL 선택 파일은 **로컬 변환 모델**을 사용한다. 이 파일만으로 다운로드되지는 않는다.
가중치 위치는 `~/.cache/huggingface/edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD`다.
[변환 방법](../weights_override/model_adapters/qwen38_lil_huihui/README.md)과
[모델 카드](../weights_override/model_adapters/qwen38_lil_huihui/docs/MODEL_CARD.md)를 참고한다.

`SPARKTALK_HF_CACHE`를 지정하면 해당 경로를 캐시 루트로 사용한다.
선택 파일은 가중치 경로와 API 모델 이름을 함께 바꾼다.

## 실행

| 구성 | 컨텍스트 | 동시 요청 | MTP |
| --- | --- | --- | --- |
| 기본 `compose.yaml` | 64K | 2 | 사용 |
| `compose.context1m.yaml` 추가 | 1M | 1 | 사용 · 단독 LLM 시험용 |
| `compose.context1m.shared.yaml` 추가 | 1M | 1 | 끔 · Flux/ASR/TTS 동시 사용용 |

**기본 64K — LIL 원본:**

```sh
docker compose --env-file model.official.env up -d
```

**1M — Huihui/LIL, Flux/ASR/TTS와 함께 사용:**

```sh
docker compose --env-file model.huihui-lil.env \
  -f compose.yaml -f compose.context1m.shared.yaml up -d
```

두 1M 파일은 서로 대체하는 설정이므로 함께 넣지 않는다.
다른 모델은 `--env-file`만 바꾼다. 기본 구성으로 돌아가려면 추가 `-f` 없이 실행한다.

동시 사용 구성은 Qwen의 KV 할당 후 Flux/ASR/TTS를 별도로 시작한다.
이 Compose는 Qwen만 실행한다. `/server_info`의 `max_total_num_tokens`가
`1048576` 이상인지 확인한다. 컨텍스트 상한만 1M인 것과 실제 KV 용량은 다르다.

## 확인 및 종료

```sh
docker compose logs -f sglang
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/server_info
docker compose down
```

API 요청의 `model`에는 선택한 모델 ID를 넣는다. 첫 기동은 적재·컴파일로 수 분 걸린다.
컴파일 캐시는 `~/.local/share/sparktalk/cache/sglang-flash-next-qad/`에 유지된다.

기본 MTP 설정은 한국어 중심 초안 어휘 `ko64k`를 사용한다. 전체 어휘를 쓰려면:

```sh
SPARKTALK_FLASH_NEXT_DRAFT_VOCAB=off docker compose --env-file model.official.env up -d
```

1M 동시 사용 구성은 MTP와 초안 어휘 제한을 모두 끈다.

