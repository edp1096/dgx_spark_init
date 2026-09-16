# 웹 사진을 문서에 넣기

사용자가 사진 검색·삽입을 요청하면 `web_search` → `web_fetch` →
`media_import` → `document_generate` 순서로 처리한다. URL 재입력은 필요 없다.

- `web_fetch`는 본문과 함께 `images[]`를 반환한다. HTML의 img src/data-src,
  og:image/twitter:image, 원본 이미지 링크를 추출하고 상대 주소를 최종 페이지 주소에 맞춰 해석한다.
- `media_import`는 사용자 제공 URL 외에, 대화 내 실제 web_search/web_fetch/web_collect
  호출 ID와 연결된 도구 결과의 구조화된 URL을 인정한다. 모델 답변, 검색 스니펫,
  페이지 본문의 임의 URL은 인정하지 않는다. 기록이 문맥에서 사라졌다면 다시 조회해야 한다.
- 웹에서 발견한 이미지는 공개 IP에 직접 연결하는 전용 HTTP 전송 경로로 다운로드한다.
  DNS 조회 후 검증된 IP로 연결하고 리다이렉트마다 목적지를 검사한다.
  내부망/루프백/링크로컬 주소, 과도한 리다이렉트, 크기 초과, 비이미지 응답을 거부한다.
  저장 시 파일 시그니처·이미지 크기도 검증한다. PNG/JPEG/WebP를 지원한다.
- 설명 페이지 URL이면 web_fetch의 이미지 후보를 선택해야 한다. 페이지를 이미지로 가져오면
  정확한 다음 동작을 오류로 안내한다. 원본 주소를 추측해 만들지 않는다.
- 반환값은 attachment.id, source_url, source_page_url, final_url을 포함한다.
  새 첨부는 대화에 저장되며 기존 문서 이미지 처리 경로로 PPTX 등에 포함된다.
  출처 확인은 사용권 확인을 대신하지 않는다. 저작자/라이선스가 확인되면 함께 표기한다.
- 사용자가 제공한 음성·영상 URL의 기존 Extra Media 처리 경로는 유지한다.

## 검증 (2026-09-15)

- webtools/server/config/orchestrator/llm 회귀 검사 통과.
- 실제 검색으로 숭례문 Wikimedia 사진 발견 → 페이지 이미지 추출 → 사용자 URL 입력 없이
  media_import → 실제 DocMS PPTX/PDF 생성 통과.
- PPTX ZIP의 ppt/media 이미지 포함 검사와 PDF 첫 페이지 사진 표시를 확인했다.
- 증거: `validation/web-images-2026-09-15/`.
- 실제 산출물: `/tmp/talk-web-image-validation/web_photo_validation.pptx`,
  `/tmp/talk-web-image-validation/web_photo_validation_preview.pdf`.
- 실행 중인 Talk 바이너리 교체·재시작 및 8585 API 정상 응답 확인.
- LLM이 이 절차를 자율 선택하는 품질은 별도이며, 이번 검증은 실제 도구 처리와 문서 생성 경로를 검증한다.

실측 재실행:

```sh
SPARKTALK_WEB_IMAGE_TEST_OUT=/tmp/talk-web-image-validation go test ./internal/server -run '^TestWebImagePPTXLive$' -v
```

Talk 프로젝트에서 실행하며 공개 웹 연결과 localhost:8696 DocMS가 필요하다.
