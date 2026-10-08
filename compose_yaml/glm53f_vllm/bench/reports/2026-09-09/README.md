# GLM 업데이트 검증 — 적용 보류

고정 DFlash7 기준은 24개 요청을 완료했다. 내용·도구 인자 검사 18개는 모두 맞았고,
강제 호출 종료 사유 3건은 `stop`이라 엄격 검사는 15/18이다. 서술형 6건은 모두
768토큰 한도에 도달했다(합산 22.54 tok/s). 이 수치는 완료 답변의 품질 점수가 아니다.

가변 길이 후보는 Entrpi DFlash2 speculator의 명시적 지원 제한으로 엔진 초기화에서
실패했고 응답은 0개다. 성능 비교값은 없으며 기본 구성에 반영하지 않았다.
공통 스케줄러의 옵션 존재만으로 개별 DFlash2 구현의 지원을 판단하면 안 된다.
MiaAI E3도 현재 B12X planned Trellis 구현의 교체용 커널이 아니므로 설치하지 않았다.

- [기준 응답](baseline/responses.json), [집계](summary.json)
- [실패 상태](adaptive-failed/attempt-status.json), [엔진 로그](adaptive-failed/head-server.log)
- [검토 및 재현 조건](../../README.md), [시험 패치](../../adaptive-attempt.patch)

상한 128K·8 GiB KV·슬롯 2로 순차 측정했다. 524K 운영 프로파일을 재검증한 결과가 아니다.
두 실행에서 호스트 재부팅은 없었으며, 종료 후 워커의 원래 환경 파일을 복원했다.
