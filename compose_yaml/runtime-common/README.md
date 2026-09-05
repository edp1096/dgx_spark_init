# 공통 모델 관리 CLI

4개 레시피(GLM 5.3 Flash, DeepSeek V4 Vision Exp, Qwen 27B EXL3,
Flash-Next EXL3)의 공개 진입점은 `manage.sh`다.

- `manage.sh`: 공통 명령·옵션 파싱
- `runtime.sh`: 모델별 실행 어댑터
- `models.sh`: 모델 다운로드·패치·동기화
- `env.sample`: 설정 예제
- `.env`: 사용자 설정 (`MODEL_VARIANT=official|abliterated`)

```bash
./manage.sh setup --abliterated --ask-token
./manage.sh model --official
./manage.sh start
./manage.sh logs worker  # 클러스터에서만 지원
```

명령: `setup image model start stop restart status logs validate`.
`build`는 `image`, `prepare`는 `model`의 호환 별칭이다.
`setup`은 이미지와 모델 준비를 포함하지만 서버를 기동하지 않는다.
`--official`, `--abliterated`, `--ask-token`은 setup/model에서만 사용한다.
토큰 값은 HF_TOKEN 환경변수 또는 숨김 입력으로 받고 자동 저장하지 않는다.

GLM은 원본+o_proj 이식, Flash-Next EXL3는 방향 파일+runtime 강도,
DeepSeek는 별도 체크포인트와 동일 blob 하드링크로 대응한다.
Qwen 27B EXL3는 현재 Uncensored 모델만 설정돼 있어 --official은 오류다.
실행 중인 모델을 중지한 뒤 종류를 변경하고 다시 시작한다.

각 레시피의 manage.sh는 이 디렉터리의 공통 파일과 동일한 사본이다.
SparkTalk는 독립적인 내장 패키지를 사용하며 이 파일을 실행하지 않는다.
