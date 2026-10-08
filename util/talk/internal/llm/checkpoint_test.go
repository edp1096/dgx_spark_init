package llm

import (
	"strings"
	"testing"
)

func TestCheckpointRejectsTruncationAndMissingSections(t *testing.T) {
	good := ""
	for _, h := range []string{"Objective", "Decisions", "Constraints", "Facts", "Artifacts", "Completed", "Unresolved", "Next Steps"} {
		good += "## " + h + "\nNone.\n"
	}
	if err := validateCheckpoint(good, "stop"); err != nil {
		t.Fatal(err)
	}
	if err := validateCheckpoint(strings.Replace(good, "## Facts\nNone.", "## Facts", 1), "stop"); err == nil {
		t.Fatal("empty section accepted")
	}
	if err := validateCheckpoint(good, "length"); err == nil {
		t.Fatal("truncated checkpoint accepted")
	}
	if err := validateCheckpoint(strings.Replace(good, "## Constraints", "Constraints", 1), "stop"); err == nil {
		t.Fatal("missing required section accepted")
	}
}
