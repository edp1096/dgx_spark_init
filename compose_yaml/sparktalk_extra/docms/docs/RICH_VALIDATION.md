# 문서 확장 검증 — 2026-09-15

Extra Documents **0.5.0**을 Compose·Talk에 반영했다. 실행 중인 렌더러·스키마·패키지 파일의 SHA-256이 정식 소스와 일치한다. Talk와 문서 서비스, 기존 LLM의 health가 모두 정상이다.

## 실측

실제 배포 API에 6개 형식 × 3회, **18회 모두 성공**했다. Office/Hancom 원본과 별도 PDF를 모두 포함한 시간이다. 모든 PDF에서 마지막 검증 문구를 확인했다. PPTX는 50행 표를 여러 장에 나눴고, XLSX는 100행 누적 수식과 차트를 생성했다.

| 형식 | 중앙값 | 성공 |
|---|---:|---:|
| DOCX | 0.298초 | 3/3 |
| HWP | 0.610초 | 3/3 |
| HWPX | 0.609초 | 3/3 |
| PDF | 0.283초 | 3/3 |
| PPTX | 0.307초 | 3/3 |
| XLSX | 0.393초 | 3/3 |

컨테이너 제한은 CPU 2개·메모리 1GiB다. 배포 후 검사 중 cgroup peak는 **220.8MiB**, OOM·메모리 한도 초과 0회였다. 이 값은 위 샘플의 실측이며 모든 문서의 최대 소모량을 뜻하지 않는다.

## 정확도·호환성

- 기존·확장 렌더러 **33개 테스트 통과**. 표·병합·이미지·부분 서식·수식·차트 6종·피벗·목록·각주/미주·목차·글상자·오디오·인쇄/보호 옵션 등을 검사했다.
- Office 샘플 **12개를 LibreOffice에서 실제 열고 PDF로 출력한 뒤 다시 저장**했다. 목차는 실제 문서 인덱스, 차트·피벗은 실제 객체로 읽히는지 검사했다.
- HWP/HWPX 샘플 **4개 독립 파서 검사 통과**. 본문·병합 표·각주/미주·글상자 내용, 쪽 나눔 이후 본문을 확인했다.
- Talk 서버·설정·내장 소스 Go 테스트 통과. 실제 배포 서비스에 Talk 도구 호출을 보내 PPTX 표+대화 첨부 이미지가 실제 객체로 저장되고 원본/PDF 두 첨부가 반환되는 것까지 통과했다.
- DOCX/PPTX를 이미지로 렌더링해 표·목차·수식·이미지 배치를 확인했다. DOCX 목차가 입력 필드로 읽히던 문제와 HWP 쪽 번호의 잘못된 제어문자 저장을 수정했다.

Microsoft Office·한컴 프로그램 자체는 실행하지 않았다. PDF는 별도 정적 렌더링이므로 원본과 페이지 배치가 다를 수 있으며, 대화형 요소는 정적으로 표현한다. 자세한 형식별 범위는 [기능 규격](RICH_DOCUMENTS.md)을 참고한다.

## 기록·샘플

[HTTP 실측](../validation/results/2026-09-15-rich/http-results.json) · [렌더러 검사](../validation/results/2026-09-15-rich/renderer-tests.txt) · [Office 재열기](../validation/results/2026-09-15-rich/office-check.json) · [한글 독립 파서](../validation/results/2026-09-15-rich/hancom-independent.json) · [배포 상태](../validation/results/2026-09-15-rich/deployment.json)

[PPTX 표 샘플](../validation/results/2026-09-15-rich/samples/pptx/document.pptx) · [DOCX](../validation/results/2026-09-15-rich/samples/docx/document.docx) · [XLSX](../validation/results/2026-09-15-rich/samples/xlsx/document.xlsx) · [HWP](../validation/results/2026-09-15-rich/samples/hwp/document.hwp) · [HWPX](../validation/results/2026-09-15-rich/samples/hwpx/document.hwpx)
