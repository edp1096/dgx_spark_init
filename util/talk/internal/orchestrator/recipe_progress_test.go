package orchestrator

import (
	"strings"
	"testing"
)

func TestRecipeOutputRedactsSplitSecretsAndCarriageReturns(t *testing.T) {
	var lines []string
	w := &recipeOutput{token: "hf_private_token", report: func(s string) { lines = append(lines, s) }}
	w.Write([]byte("download hf_priv"))
	w.Write([]byte("ate_token\r50%\nfinished"))
	tail := w.finish()
	if strings.Contains(tail, "hf_private") || len(lines) != 3 || lines[0] != "download [redacted]" || lines[1] != "50%" {
		t.Fatalf("unexpected output: %q", lines)
	}
}
