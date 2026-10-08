# 셀렉터·생각 과정 펼침 (2026-09-16)

셀렉터는 `Select.svelte`의 버튼·목록 컴포넌트로 통일했다.
네이티브 select의 pointerdown/mousedown을 차단하거나 showPicker()를 호출하지 않는다.
HTML popover를 사용하는 목록은 별도 항목 클릭으로 선택되며, 폼 높이에 영향을 주지 않는다.
키보드 이동·선택·Escape, 현재 선택 강조, 모바일 화면 배치를 지원한다.
기존 option과 값 바인딩은 비표시 select로 연결하고, 화면 제거 시 함께 정리한다.

생각 과정은 details의 비동기 toggle 이벤트 대신 Svelte 상태 한 곳에서 펼침을 관리한다.
스트리밍 갱신·응답 저장 도중에도 사용자의 펼침 상태를 유지한다.

Markdown의 단일 물결표는 그대로 표시하며, 취소선은 두 개의 물결표만 사용한다.

검증: 실제 Chromium/X11 마우스 누름·이동·해제, 반복 선택,
모바일 목록 배치, 설정 저장·조합 편집, 스트리밍 중 반복 펼침·접기.
관련 테스트는 web/e2e/select-on-release.spec.js, reasoning-toggle.spec.js,
context-presets.spec.js, src/lib/markdown.test.js에 있다.

실행본 반영: 2026-09-16 03:39 KST. 프런트엔드 단위 테스트 92개 통과.
전체 브라우저 테스트 69/70 통과 후 모바일 스크롤 단독 재검증 2/2 통과.
초기 대화 복원이 늦게 완료되어도 사용자가 연 라이브러리를 닫지 않도록 수정했다.
