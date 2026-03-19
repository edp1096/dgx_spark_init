# ZIT 참고 사항

## 아키텍처

```
ZIT Gradio UI (app.py)
├── tab_generate.py    — T2I 생성 (최대 5 LoRA 동시)
├── tab_inpaint.py     — Inpaint/Outpaint (gr.Tab 서브탭)
├── tab_history.py     — 생성 이력 갤러리
├── tab_train.py       — LoRA 학습 (aspect-ratio bucketing)
├── tab_settings.py    — 설정 (gr.Sidebar TOC)
├── generators.py      — 생성 요청 → worker 제출
├── worker.py          — 백그라운드 파이프라인 실행
├── pipeline_manager.py— 모델 로딩/언로딩
├── i18n.py            — MutationObserver 기반 다국어
├── helpers.py         — lora_choices, memory_status 등
└── zit_config.py      — 경로/상수 설정
```

## 배포 워크플로우
1. `/root/zit-ui/app/` → `/root/dgx_spark_init/compose_yaml/zit/app/` 복사
2. git commit + push (dgx_spark_init 레포)
3. 컨테이너 재시작 시 GitHub API로 코드 동기화

## Gradio 핵심 주의사항
- `visible=False` → DOM unmount (PixiJS 파괴) → gr.Tab 사용
- `every=N` → queue에서 ImageEditor 상태 리셋 → gr.Timer + gr.update() 사용
- sticky/fixed CSS 불가 (overflow 체인) → gr.Sidebar 사용
- 앱이 PID 1 — 에러 = 컨테이너 사망. py_compile + import 검증 필수

## LoRA 현황
- Multi-LoRA: 최대 5개 동시 탑재, 개별 scale
- 학습: ostris ai-toolkit 기반, aspect-ratio bucketing 17종
- 다운로드: HuggingFace/CivitAI/직접URL 통합 입력
- 업로드: .safetensors 직접 업로드, rank/alpha 자동 추출
- 메타데이터: lora_metadata.json (trigger_words, recommend_scale 등)

## 모델 정보
- Turbo + training adapter: ostris/zimage_turbo_training_adapter (v2)
- ControlNet Union lite: Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite (2.02 GB)
- SCRFD: cv2.dnn 백엔드, preprocessors/ 폴더
