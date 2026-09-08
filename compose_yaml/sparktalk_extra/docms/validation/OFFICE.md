# Office 샘플 검증

실행 중인 Extra Documents의 이미지 ID를 고정하고, 실제 API에서 DOCX·PPTX·XLSX·PDF 샘플을 생성합니다.

```sh
# docms 폴더에서
./validation/check-office.sh
```

- DOCX: 모든 문단과 81행 × 2열 표를 원본·LibreOffice 재저장본에서 비교.
- PPTX: 두 슬라이드의 제목·목록·페이지 번호를 원본·재저장본에서 비교.
- XLSX: 모든 셀, 수식 10개와 결과의 자료형, 날짜·금액 서식, 행 고정·필터·인쇄 설정을 비교.
- PDF: 전체 내용과 행 순서, 모든 페이지의 잘림·글자 겹침·폰트 포함·머리글 대비를 검사.
- 누락·오계산·자료형 변형·잘림을 일부러 넣은 대조군이 실패하는지도 확인.

재열기는 가상 화면의 실제 LibreOffice 컨트롤러로 수행합니다. `--headless --convert-to`는 별도로 만든 기준 XLSX에서도 행 고정을 잃어 이 검사에 사용하지 않습니다.

결과: [전체 페이지 보고서](../runs/samples/verification.html), [검사 결과·버전·파일 SHA-256](../runs/samples/verification.json). 생성물은 Git에서 제외됩니다.

이 네 샘플의 내용·값·표시 검증이며, 임의 문서 전체나 Microsoft Office 호환성 인증은 아닙니다. Office와 별도 PDF의 페이지 배치가 동일해야 한다는 조건은 적용하지 않습니다.

2026-09-07, Extra Documents 0.3.1: 자동 검사 13건·PDF 19페이지 시각 검사·손상 대조군 9건 모두 통과했습니다. 자동 필터의 범위 이름 누락을 수정해 원본과 LibreOffice 재저장본에서 필터 보존을 확인했습니다. 일반 숫자 형식의 `300`/`300.00` 표시는 같은 값으로 확인하고 보고서에 기록했습니다.
