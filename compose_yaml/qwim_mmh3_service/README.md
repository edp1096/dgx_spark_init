# QWIM + MiniMax H3 서비스

Talk의 **Qwen 3.8 Flash-Next EXL3** 세트에 쓰는 공동 상주 엔진이다.
검증된 QWIM/H3 DiT를 한 GPU 프로세스에 유지하고, 텍스트 인코더와 VAE는 단계 뒤
해제한다. LLM/ASR/TTS는 기존 서비스를 재사용한다. 다른 QWIM 이미지 편집 세트로 전환하면 두 이미지 엔진을 교체한다.

- 이미지: 텍스트 입력, 1024×1024, 40스텝, `/v1/images/generations`.
- 영상: 텍스트 입력은 864×480, 이미지 입력은 원본 비율에 맞춘 자동 해상도
  (최대 픽셀 면적 864×480). 124프레임/24fps(약 5.17초), 20스텝, AAC 음성 포함 MP4,
  `/v1/videos/generations`. `first_frame`·`last_frame`에 PNG/JPEG/WebP data URL을
  전달하면 실제 0·123프레임 조건으로 인코딩한다. 이미지당 32MiB·16메가픽셀 한도다.
  Talk의 `video_generate`는 `first_frame_image_id`·`last_frame_image_id`로 현재 대화의
  첨부 이미지를 선택한다. 첫 프레임은 VAE 조건이므로 원본 픽셀과 완전히 같지는 않다.
  임의 해상도·길이, 참조 영상·오디오 입력은 제공하지 않는다.
- 영상은 SM121 GPU의 Sol-Attn 단독을 사용한다. 처음 4·마지막 2스텝과 처음 2블록,
  텍스트·오디오 query는 dense 계산을 유지한다. Spectrum·FBC는 사용하지 않는다.
  같은 시드라도 dense 원본과 구도·동작이 달라질 수 있다. `MMH3_ATTENTION=dense`로
  되돌릴 수 있으며 이미지 생성에는 이 설정이 적용되지 않는다.
- `/health`는 두 DiT 적재가 끝난 뒤 정상으로 응답한다.
- 작업은 직렬화하며, `/v1/runtime/memory`는 실행·대기 작업을 함께 보고한다.
- `/v1/runtime/quiesce`는 실행·대기 작업이 있으면 409를 반환한다.
- 취소·시간 초과는 해당 워커만 종료한다. 생성 파일은 HTTP 전달 뒤 임시 보관소에서 삭제한다.

진행 표시에는 현재 단계·스텝·경과 시간과 예상 남은 시간이 나온다. 샘플링 속도로
추정치를 보정하며, 최근 완료 작업 8회의 단계별 시간으로 전체 ETA를 계산한다.
기록이 부족하면 현재 단계 ETA만 표시하고 전체 시간은 계산 중으로 둔다. 진행 로그는
완료 후 대화에 보관하며, 단계별 시간 기록은 전용 Docker 볼륨에 유지한다.

8개 체크포인트의 고정 리비전과 SHA-256은 `prepare_models.py`에 있다.
Hugging Face 캐시는 읽기 전용으로 마운트하며, 이미 있는 체크포인트는 재사용한다.
Talk의 모델 준비 화면은 필요한 모델을 내려받을 수 있다. Docker 빌드 입력과
ClipProj의 MIT 라이선스를 실행 파일에 포함하므로 앱 배포 후 소스 체크아웃은 필요 없다.

```sh
docker compose build
docker compose up -d
```

36GiB cgroup 상한과 28GiB 작업 입장 예산, 최소 6GiB 작업 공간은 검증 프로필의 한도다.
모델 파일 크기와 실제 실행 중 최대 메모리는 다르며, Talk은 남은 시스템 메모리를
검사한 뒤 작업을 시작한다. 이 서비스가 지원하는 작업 범위는 위 두 고정 프로필이다.
