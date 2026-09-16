package server

import (
	"strings"
	"testing"
)

func TestInternalEvidenceOnly(t *testing.T) {
	marker := `[Historical tool evidence: web_fetch; archive_id=524; use context_read for original] {"content":"private context"}`
	input := "답변\n" + marker + "\n다음 설명\n" + `{"valid":"JSON"}` + "\n[일반 대괄호]"
	out := cleanInternalEvidence(input)
	if strings.Contains(out, "private context") || !strings.Contains(out, `{"valid":"JSON"}`) || !strings.Contains(out, "다음 설명") || !strings.Contains(out, "[일반 대괄호]") {
		t.Fatal(out)
	}
	code := "```text\n" + marker + "\n```"
	if cleanInternalEvidence(code) != code {
		t.Fatal("code example changed")
	}
}
