# Tool Eval Bench

실행 중인 모델 API를 평가하는 독립 Compose입니다. SparkTalk·Extra와 별개이며 모델 서버를 시작하거나 중지하지 않습니다.

## 준비

```bash
./manage.sh setup
# .env에서 TOOL_EVAL_BASE_URL 지정. 필요하면 MODEL/API_KEY도 지정.
./manage.sh probe
```

원본을 `upstream/`에 클론하고 `upstream.env`의 커밋으로 고정해 빌드합니다.

## 실행

```bash
./manage.sh run --short --seed 42  # 15문항, 전경 실행
./manage.sh start                 # 88문항 × 3회, 백그라운드 실행
./manage.sh logs                  # 진행 확인
./manage.sh status                # 실행 상태
./manage.sh stop                  # 중단
```

추가 옵션을 주면 기본 옵션을 대체합니다. `logs`에서 Ctrl-C는 로그 보기만 종료합니다.

## 결과

```bash
./manage.sh history               # 실행 이력
./manage.sh reports               # 모델·점수·시간 목록, 읽기 쉬운 파일명 생성
./manage.sh compare               # 목록에서 두 결과 선택 → HTML 생성
./manage.sh compare runs/named/A.md runs/named/B.md  # 파일 직접 지정도 가능
./manage.sh cli resume RUN_ID     # 중단한 평가 이어하기
```

- `runs/named/`: 모델명_UTC시각_trial1.md 형태의 영문 파일명 (원본 연결)
- `runs/`: 원본 점수표와 대화 기록. 재개·이력용 원본은 유지합니다.
- `data/benchmarks.sqlite`: 이력·재개용 DB
- `cache/`: 다운로드 캐시

회차 번호는 전체 요약이 저장되면 붙습니다. 진행 중에는 `run`으로 표시합니다.
`run`과 `start` 모두 평가 종료 후 링크를 자동 생성합니다.
`reports`와 `compare`도 조회할 때 새 결과를 반영합니다. 호스트 Python 3을 사용합니다.

점수와 함께 완료율을 확인하고, 같은 평가 버전·설정끼리 비교하세요.

전체 옵션: `./manage.sh cli --help` · [원본 저장소](https://github.com/SeraphimSerapis/tool-eval-bench)
