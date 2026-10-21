# Compose 설정

`env.sample`이 있는 구성은 해당 디렉터리에서 `cp -n env.sample .env` 후 값을 수정한다. `.env`는 Git에서 제외하고 `env.sample`은 커밋한다. 토큰·비밀번호는 예제에 넣지 않는다.

`manage.sh`가 있는 모델은 스크립트가 `.env`를 자동 생성한다. 직접 Compose를 쓰는 구성은 같은 디렉터리의 `.env`를 자동으로 읽는다. 셸 스크립트를 직접 실행할 때는 `set -a; . ./.env; set +a`로 적용한다.

llama.cpp의 systemd 실행은 `env.sample`을 기존 이름인 `llama-server.env`로 복사한다. 별도 설정 변수가 없는 Compose는 `.env`가 필요 없다.

The QAD and EXL3 4bit recipes are archived under `zzz_not_use/qwen38_fn_qad` and `zzz_not_use/qwen38fn_exl3_q4`. The retired TP1 RadixArk recipe is under `zzz_not_use/qwen38_fn_radixark_nvfp4`; shared TP2 files remain in `qwen38_fn_sglang`.

Qwen 27B EXL3는
`zzz_not_use/qwen38_27b_exl3`, GGUF 구성과 검사 기록은
`zzz_not_use/qwen38_27b_gsq_rco_gguf`, NVFP4/SGLang은
`zzz_not_use/sglang_qwen38_27b`에 보관한다. 보관한 27B 구성은 SparkTalk 모델 목록에 포함하지 않는다.

Huihui v2 FP4-Local은 [qwen38_fn_fp4_local_tsfold](qwen38_fn_fp4_local_tsfold/README.md)에 있다.
SparkTalk에서는 `Qwen 3.8 Flash-Next FP4-Local` 세트로 선택한다.

Qwen EXL3 3bit uses [TensorFold](qwen38fn_exl3_tsfold/README.md). The previous vLLM and ExLlamaV3 recipes are archived under `zzz_not_use`.

The TP1 RadixArk NVFP4 recipe is archived under `zzz_not_use/qwen38_fn_radixark_nvfp4`. TP2 remains in `qwen38_fn_sglang` and still shares its model weights.

GLM 5.3 Flash uses [FP4-Local / vLLM B12X TP2](glm53f_fp4_local_vllm/README.md), with abliterated weights, FP8 KV, 1M context and MTP 5 by default. The previous NVFP4 SGLang and older vLLM recipes are archived under `zzz_not_use/glm53f_sglang` and `zzz_not_use/glm53f_vllm`. The experimental FP4-Local TensorFold build is archived under `zzz_not_use/glm53f_fp4_local_tsfold`; it is not the selected Talk runtime.

Source: local matched comparison and full-context qualification, linked from the active GLM recipe.

Gemma 4 31B: [NVFP4 + EXL3 assistant / SGLang](gemma4_31b_sglang/README.md), fixed MTP9 with Split-KV verify. Talk set: `Gemma 4 31B NVFP4`.
