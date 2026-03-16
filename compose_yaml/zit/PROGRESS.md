# ZIT 진행 상태

## 완료된 작업

### Phase 1~4, 6~7: 기본 구현 완료
- zifk 복사 → Klein/FLUX 전부 제거
- VideoX-Fun 모델 클래스 통합 (videox_models/)
- ControlNet + Inpaint/Outpaint 핸들러
- 전처리기 (Canny, DWPose, ZoeDepth, HED, Gray)
- Gradio UI 6탭 (Generate, ControlNet, Inpaint, FaceSwap, Settings, History)
- Builder/Deploy (compose.yml, setup.sh, update.sh, compose.yaml)

### DGX 테스트 결과
- ZIT T2I 생성: 동작 확인
- 다중 이미지 생성: `control_context` 배치 복제 버그 수정 → 동작 확인
- Gradio 6.0 호환: `show_share_button` → `buttons=["download","fullscreen"]` 수정
- update.sh 재귀 탐색: zoe/ 하위 트리 다운로드 수정
- update.sh GitHub API rate limit: GITHUB_TOKEN 인증 추가

---

## Phase 5 변경: FaceSwap 방식 전환 (2026-03-16)

### 기존 (실패)
```
SCRFD(TRT) → ArcFace(TRT) → inswapper(TRT) → CodeFormer → paste back
```
- TRT 엔진이 Blackwell(GB10)에서 깨진 출력 (마네킹 얼굴)
- 입출력 정규화 문제도 복합적으로 존재

### 변경 (ComfyUI Z-Image Headswap 방식)
```
SCRFD(cv2.dnn) → 자동 마스크 생성 → ZIT Inpaint → 얼굴 프롬프트 기반 재생성
```
- inswapper/ArcFace/CodeFormer **전부 제거**
- SCRFD는 cv2.dnn 백엔드 (DWPose와 동일, aarch64 검증)
- ZIT Inpaint 파이프라인 재활용

### 호스트에서 수정 완료된 파일
- `zit_config.py` — FACESWAP_DIR/ARCFACE/INSWAPPER/CODEFORMER 삭제, SCRFD_URL 추가
- `face_swap.py` — 완전 재작성 (SCRFDDetector cv2.dnn + create_face_mask + preview)
- `download_models.py` — ArcFace/inswapper/CodeFormer 다운로드 제거, SCRFD만

### 컨테이너에서 수정 필요한 파일
→ `CONTAINER_TODO.md` 참조
- `app.py` — FaceSwap 탭 UI 변경 (target+source → image+prompt+자동마스크)
- `generators.py` — generate_faceswap → SCRFD 마스크 + inpaint 호출
- `worker.py` — _run_faceswap → create_face_mask + _run_inpaint 위임
- `pipeline_manager.py` — load_faceswap → SCRFD cv2.dnn 로딩
- 기존 faceswap/ 폴더 및 .engine 파일 정리
