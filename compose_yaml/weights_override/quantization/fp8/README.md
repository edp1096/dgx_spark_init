# FP8 변경분 이식

기존 Qwen3.5 등의 FP8 변환도 루트 `convert.py`로 실행한다.

```bash
python3 convert.py --profile fp8 \
  --original /models/original-bf16 --donor /models/ablit-bf16 \
  --base /models/original-fp8 --output /models/ablit-fp8 \
  --activation-scales preserve
```

기존 방식인 `복원한 FP8 베이스 + (ablit − 원본)`을 유지한다.
이 폴더의 기존 스크립트·Docker 환경은 비교 검증용으로 보존한다.
