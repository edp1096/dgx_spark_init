# DocMS

문서 생성·수식 계산·PDF 출력을 담당하는 SparkTalk Extra 서비스입니다. DOCX·PPTX·XLSX·HWP·HWPX·PDF를 지원합니다. Office·한글 파일에는 같은 내용으로 별도 생성한 PDF도 반환하며, PDF만 실패하면 원본과 `warning`을 반환합니다. GPU와 LibreOffice 없이 동작합니다.

```sh
cp env.sample .env
./manage.sh start
./manage.sh status
```

SparkTalk 설정 → 기능 → 문서에서 활성화하고 API 주소를 입력합니다(기본 `http://127.0.0.1:8696`). 다른 서버에서 접근하려면 `.env`의 바인드 주소를 해당 LAN 주소로 설정합니다.

`POST /v1/documents`: `format`, `title`, `sections`(heading/paragraphs/table/images) 또는 `slides`(title/bullets), XLSX는 `sheets`(name/columns/rows). `GET /health`: 지원 형식·작업 상태. 작업은 동시에 1개, 최대 90초입니다. 파일명은 ASCII이며 응답의 `files`에 base64 파일이 포함됩니다.

본문·표, 목록형 슬라이드, 여러 시트와 기본 수식을 지원합니다. DOCX·PDF·HWP·HWPX에는 세션에 첨부된 이미지를 넣을 수 있습니다. SparkTalk 도구는 `images: [{image_id, width_cm, caption}]`으로 기존 이미지 ID를 참조하며, 서버가 PNG로 변환해 전달합니다(최대 6개, 변환 후 합계 8 MiB). 외부 URL·임의 HTML·스크립트는 받지 않습니다. PDF는 Office 원본과 줄바꿈·페이지 배치가 다를 수 있습니다.

SparkTalk의 빌드 입력도 바이너리에 포함됩니다. 이 폴더 없이 별도로 기동할 수 있습니다.

한글 문서 검증은 `./validation/check-hancom.sh`로 재현합니다. 패치된 rhwp 0.8.6으로 본문·표·이미지를 생성하고 수정·재저장, 독립 파싱을 검사합니다. 생성된 한글 파일의 본문은 SparkTalk에 보관되어 후속 대화에서 참조할 수 있습니다. 임의 HWP 업로드 분석은 이번 범위에 포함하지 않습니다. 실제 한컴에서 열기·편집·저장은 별도 확인 대상입니다.

검증 결과와 독립 파서 재현 명령은 [validation/README.md](validation/README.md)를 참고하세요. HWP 표 헤더와 새 문단의 불필요한 페이지 나눔은 `hwp-engine/patch.py`에서 수정합니다.

XLSX는 최대 8시트·시트당 1,000행·32열·전체 20,000셀입니다(요청 본문 제한도 적용). 1행은 머리글, 데이터는 2행부터 시작합니다. 열의 `format`은 general/number/integer/currency/percent/date, `width`는 6–60입니다. `freeze_header`·`filter`는 기본 true, `orientation`은 portrait/landscape입니다. 숫자·문자·불리언·null, 날짜 `{"date":"2026-09-07"}`, 수식 `{"formula":"SUM(B2:B3)"}`를 셀 값으로 사용합니다. 파일명은 영문, 시트명과 내용은 한글을 지원합니다.

수식은 기본 연산과 SUM/AVERAGE/COUNT/COUNTA/MIN/MAX/IF/AND/OR/NOT/ROUND/ROUNDUP/ROUNDDOWN/ABS/COUNTIF/SUMIF/IFERROR를 지원합니다. 제공된 셀만 참조하며 순환 참조와 계산 오류는 거부합니다. Excelize로 계산한 결과를 원래 수식·서식과 함께 저장합니다. 차트·피벗·매크로는 미지원입니다. Excel 자체에서의 호환성은 별도 확인 대상입니다.

AND/OR/NOT에는 범위 대신 개별 셀·비교식을 사용합니다. 넓은 시트의 PDF는 열을 나누어 표시합니다. 엔진은 docx·PptxGenJS·ExcelJS·pdfmake(MIT), Excelize(BSD-3-Clause)이며 글꼴은 Noto CJK(OFL)입니다. [의존성 고지](THIRD_PARTY_NOTICES.md).

운영 이미지에는 LibreOffice가 없습니다. 개발 검증만 별도 이미지로 실행합니다:

```sh
docker build -t sparktalk-extra-documents:0.4.0 .
docker build -f validation/Dockerfile.office -t sparktalk-documents-validation .
docker run --rm sparktalk-documents-validation node --test renderer.test.mjs
mkdir -p runs/samples
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/validation:/validation:ro" -v "$PWD/runs/samples:/out" sparktalk-documents-validation node /validation/office-roundtrip.mjs /out
```

전체 샘플 대조는 `./validation/check-office.sh`로 실행합니다. [검사 범위와 결과](validation/OFFICE.md)를 참고하세요.
