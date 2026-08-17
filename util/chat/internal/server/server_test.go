package server

import "testing"

func TestFallbackTitle(t *testing.T) {
	if got := fallbackTitle("  간단한   테스트 제목  "); got != "간단한 테스트 제목" {
		t.Fatalf("unexpected title: %q", got)
	}
	long := fallbackTitle("이 문장은 자동 제목의 최대 길이를 확실하게 넘어가도록 충분히 길게 작성한 테스트 문장입니다")
	if []rune(long)[len([]rune(long))-1] != '…' {
		t.Fatalf("long title was not shortened: %q", long)
	}
}
