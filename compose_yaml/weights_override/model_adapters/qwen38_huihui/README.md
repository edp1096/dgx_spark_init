# Qwen3.8 Huihui → RadixArk NVFP4

Huihui GGUF의 변경분을 RadixArk safetensors에 이식하는 내부 어댑터.
실행은 루트의 `convert.py --profile qwen38_huihui`로 통합했다. 아래 스크립트를 개별 실행할 필요는 없다.

| 단계 | 스크립트 |
|---|---|
| GGUF 전체 비교 | `audit.py` |
| BF16 변경분 이식 | `build.py` |
| 원본 expert 확보 | `fetch_source.py` |
| Expert 이식·NVFP4 양자화 | `patch_experts.py`, `run_quant_container.py` |
| 검증·모델 비교 | `verify.py`, `test_numeric.py`, `compare_dealign.py` |
| 실행 검증 | `launch_validation.py`, `evaluate.py`, `summarize_runtime.py`, `final_validation.py` |
| 모델 카드·업로드 | `prepare_release.py`, `publish.py` |

[변환 절차와 검증 기록](docs/README.md). 모델 가중치는 Hugging Face 캐시에 별도 보관한다.
