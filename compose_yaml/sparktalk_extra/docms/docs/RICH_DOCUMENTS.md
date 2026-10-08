# 문서 기능 확장 — 0.5.0

기존 입력은 유지한다. `sections[].blocks`와 `slides[].blocks`를 쓰면 문단·표·이미지를 원하는 순서로 배치할 수 있다. 같은 섹션/슬라이드에서 `blocks`와 기존 본문 필드는 혼합하지 않는다. 정확한 입력 규격은 [document-schema.json](../document-schema.json), 실행 서비스에서는 `GET /v1/documents/schema`로 확인한다.

| 형식 | 추가 기능 |
|---|---|
| PPTX | 실제 표·병합 셀·긴 표 분할·반복 헤더, 이미지와 자르기, 차트 6종, 도형·화살표·글상자, 두 단 배치, 테마·화면 비율, 부분 서식·링크, 발표자 노트, MP3/WAV/MP4 첨부 |
| DOCX | 부분 서식, 제목 단계·목차·실제 번호/글머리표, 머리말·꼬리말·쪽 번호, 첫/짝수 페이지 구분, 페이지·구역·다단, 병합 표·셀 서식·중첩 표·셀 안 이미지, 이미지 배치, 링크·북마크·각주·댓글·수식·체크박스 |
| XLSX | 셀 서식·부분 서식·병합, 조건부 서식·드롭다운·입력 검증, 이미지·링크·메모, 행/열 그룹·숨김·시트 숨김·고정 위치·보호, Excel 표 객체, 인쇄 옵션, 차트·피벗·도형, VLOOKUP/XLOOKUP/SUMIFS/COUNTIFS |
| HWP/HWPX | 글자·문단 서식, 번호/글머리표, 병합 표·셀 서식, 머리말·꼬리말·쪽 번호, 용지·여백·다단·쪽/구역 나눔, 이미지 배치, 각주·미주·수식·도형·글상자·북마크 |
| PDF | 순서형 본문, 표·셀 병합·부분 서식·이미지·차트·배치. PPTX 표 분할은 동일한 배치 계획을 사용한다. XLSX는 수식 계산 후 표시값과 정적 집계를 사용한다. |

## 표가 있는 PPTX

```json
{
  "format": "pptx",
  "title": "실적 보고",
  "slides": [{
    "title": "분기 실적",
    "blocks": [
      {"type": "paragraph", "text": "매출과 비용을 비교합니다."},
      {"type": "table", "rows": [["항목", "금액"], ["매출", "120"], ["비용", "80"]]}
    ]
  }]
}
```

`table`은 문자열 또는 `{text, style, col_span, row_span}` 셀의 행 배열이다. 병합으로 덮인 칸은 생략한다. 첫 행은 헤더다. DOCX/PDF 셀에는 `blocks`로 문단·이미지·중첩 표를 넣을 수 있다. PPTX의 병합 표는 한 장에 들어가야 하며, 단일 행이 슬라이드보다 길면 분할 안내 오류를 반환한다. 내용을 잘라 버리거나 표를 텍스트로 대체하지 않는다.

이미지는 Talk에서 `{image_id, width_cm, caption}`으로 기존 대화 첨부를 참조한다. 서버가 PNG로 변환하며 최대 6개·합계 8MiB다. PPTX 미디어는 `{media_id}`로 참조하며 최대 3개·합계 8MiB다. 경로·URL·사용자 직접 base64는 Talk에서 받지 않는다. 외부 웹페이지 링크는 별도 `link` 속성으로만 저장한다.

텍스트는 문자열 또는 `[{text, style, link}]` 배열이다. 색은 `264D73`처럼 6자리 HEX다. 페이지 여백과 글자 크기는 pt, 슬라이드 배치 `x/y/w/h`는 inch, XLSX 이미지 너비·높이는 px다. 표 `widths`는 상대 비율이다. 프리셋은 business/minimal/dark, 슬라이드 비율은 wide/standard다.

## 저장·검증 범위

- Office 원본은 실제 편집 가능한 객체로 저장한다. PDF는 별도 정적 출력이므로 줄바꿈·쪽 번호·배치가 원본과 다를 수 있다. 음성/영상은 PDF에서 안내 텍스트로, 각주·댓글은 인접 설명으로 표시한다. 피벗은 PDF에서 정적 집계다. 기본 PDF 글꼴은 Noto CJK다.
- DOCX 목차는 제목 링크와 갱신 가능한 실제 TOC 필드다. Office에서 필드를 갱신하면 원본의 페이지 번호를 계산한다. 생성 시 아직 계산하지 않은 원본 페이지 번호를 임의로 넣지 않는다.
- HWP/HWPX 페이지 속성은 문서 전체에 적용한다. 섹션별로는 `break_before`를 지원한다. 첫 페이지 머리말 변형은 DOCX/PDF 전용이다.
- HWP 쪽 번호는 표준 `pageNum` 컨트롤을 사용한다. HWPX를 거쳐 HWP로 내보낼 때 원본 요약 정보 스트림을 보존한다. 엔진의 잘못된 단일 제어문자 필드 출력 경로는 사용하지 않는다.
- XLSX 차트·피벗·도형은 ExcelJS 저장이 끝난 뒤 Excelize로 추가해 재저장 손실을 피한다. 병합 대상의 덮이는 셀에 내용이 있거나 피벗 위치가 원본 표와 겹치면 거부한다. 새 수식도 범위·순환 참조 검사와 실제 계산을 거친다.
- 원문 크기 160,000자, 요청 최대 32MiB, 결과 파일별 24MiB, 작업 90초·동시 1개 제한을 유지한다. Office 컨테이너 메모리는 1GiB다. 문단 개수 제한은 추가하지 않았다.

검증은 생성 파일 XML/객체 확인, PDF 텍스트 확인, LibreOffice 열기·재저장, 독립 HWP 파싱, HTTP 실측을 포함한다. **실제 Microsoft Office와 한컴 프로그램 실행 검증은 포함하지 않는다.** HWP/HWPX는 rhwp 재열기와 독립 파서로 검사했다.

재현: `node --test renderer.test.mjs rich.test.mjs`를 [검증 이미지](../validation/Dockerfile.office)에서 실행한다. 실제 API 실측은 `python3 validation/http-features.py --url http://127.0.0.1:8696 --out runs/http`로 실행한다. 앱 스키마를 변경하면 `python3 tools/build_schema.py`, Talk 내장 소스를 갱신하면 Talk 디렉터리에서 `go run ./cmd/package-support`를 실행한다.
