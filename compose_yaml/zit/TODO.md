# ZIT 다음 세션 작업

## 1. ImageEditor 브러시 문제 디버깅

Inpaint 탭 ImageEditor에서 브러시 크기 조절 등 내부 상태가 리셋되는 문제 추적 중.

### 배제된 원인
- ~~i18n MutationObserver~~ — 비활성화해도 동일 증상. 복원 완료.
- ~~커스텀 gradio whl~~ — 제거 완료. 공식 gradio 사용 중.
- ~~visible 토글~~ — gr.Tab 전환으로 교체 완료.
- ~~every=N 파라미터~~ — gr.Timer로 교체 완료.

### 남은 용의자: gr.Timer tick DOM 업데이트
현재 앱에서 지속적으로 DOM에 영향을 주는 것들:

| 대상 | 주기 | outputs |
|---|---|---|
| gr.Timer 1s ×3 | 1초 | loading_md(gen), loading_md(ip), train progress |
| gr.Timer 2s ×4 | 2초 | gen info, ip info, train status, train log |
| gr.Timer 3s ×1 | 3초 | memory_md |
| gr.Timer 10s ×2 | 10초 | history gallery, lora table |
| i18n MutationObserver | DOM변경마다 | 텍스트 번역 (ImageEditor SKIP) |
| scroll listener | 스크롤마다 | Settings TOC 하이라이트 |

### 디버깅 방법
- app.py를 백업 후 별도 실행 (PID 1이 아닌 환경)
- Timer를 하나씩 비활성화하면서 ImageEditor 동작 확인
- 특히 Inpaint 탭 활성 시 1s/2s timer의 outputs가 같은 탭 컴포넌트를 업데이트하는지 확인

## 2. ControlNet Tile 업스케일 모델 추가

- Tile 모델: `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` 레포
  - lite: `Z-Image-Turbo-Fun-Controlnet-Tile-2.1-lite-2601-8steps.safetensors` (2.02 GB)
  - full: `Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.safetensors` (6.71 GB)
- Union(멀티컨트롤)과 별개 모델 — Union으로 tile/업스케일 불가
- download_models.py에 Tile 다운로드 추가
- 파이프라인에 업스케일 모드 구현
- 아웃페인트 시 ControlNet 사용 문제도 교정 필요

## 3. FaceSwap 탭 UI 구현

SCRFD(cv2.dnn) 기반으로 전환 완료 (호스트 코드). 컨테이너에서:
- app.py — FaceSwap 탭 UI (image + prompt + 자동마스크)
- generators.py — generate_faceswap (SCRFD 마스크 + inpaint)
- worker.py — _run_faceswap (create_face_mask + _run_inpaint 위임)
- pipeline_manager.py — load_faceswap (SCRFD cv2.dnn)
- 기존 faceswap/ 폴더 및 .engine 파일 정리
