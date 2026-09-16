# PPTX 표 너비와 생성 결과 — 0.5.1

슬라이드의 `table`에 `{rows,widths,style}`을 넣거나, `blocks`에
`{type:"table",rows,widths,style}`을 넣으면 같은 경로로 처리한다.
기존 문자열 행 배열 형식도 유지한다. `widths`는 열별 상대 너비이며
병합 셀은 해당 열들의 너비를 합산한다.

```json
{"title":"비교","table":{"rows":[["구분","설명"],["항목","긴 설명"]],"widths":[1,4],"style":{"size":14}}}
```

서비스 응답의 `presentation`에는 실제 저장된 PPTX에서 읽은 `slide_count`,
표별 `slide_numbers`, `rows_per_slide`, `column_widths_inches`, `split`을 담는다.
`source_slide`는 요청 슬라이드 번호, 표 번호는 생성 계획의 식별자다.
분할된 표의 행 수에는 반복 헤더가 포함된다. 시각 검사 결과가 아니므로
`evidence: generated_pptx_structure`, `visually_verified: false`로 반환한다.

Talk는 이 정보를 도구 결과에 전달한다. 실제 다운로드 파일명은 첨부파일의
이름을 따르며, 사용자 지정 이름은 `filename`에 ASCII 파일명 줄기를 넣는다.
구조 정보만으로 파일을 눈으로 확인했다거나 가독성을 확인했다고 말하지 않도록
도구 설명과 결과에 명시했다. 모델의 지시 준수 자체를 보증하는 기능은 아니다.

검증:
- 렌더러 34개 테스트, Talk 서버·오케스트레이터·설정 테스트 통과.
- 실행 중인 서비스에서 1장 표·3장 분할 표·병합 표 생성: 약 0.30–0.33초.
- 실제 PPTX 열 너비·행 수·슬라이드 수가 반환 정보와 일치.
- 미리보기 PDF 및 LibreOffice에서 PPTX를 열어 내보낸 PDF는 각각 1·3·1쪽.
- 한 장 비교표의 미리보기와 LibreOffice 출력은 이미지로 확인.
- Talk 실제 도구 호출에서 PPTX·PDF 첨부와 구조 정보 전달 확인.
- Microsoft PowerPoint GUI에서 직접 열어보지는 않았다.

[HTTP 결과](../validation/results/2026-09-15-tables/http-results.json) ·
[배포 상태](../validation/results/2026-09-15-tables/deployment.json) ·
[한 장 표](../validation/results/2026-09-15-tables/one/document.pptx)
