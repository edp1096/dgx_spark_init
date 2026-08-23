package prompt

import (
	"strings"
	"testing"
)

func TestImageEnhancementModesHaveTaskSpecificRules(t *testing.T) {
	tests := []struct {
		mode string
		want string
	}{
		{mode: "t2i", want: "55-100 words"},
		{mode: "edit", want: "Change and Preserve"},
		{mode: "control", want: "reference image controls pose"},
		{mode: "paint", want: "AnyPaint inpainting and outpainting"},
	}
	for _, test := range tests {
		t.Run(test.mode, func(t *testing.T) {
			if got := System(test.mode, false); !strings.Contains(got, test.want) {
				t.Fatalf("System(%q) does not contain %q: %s", test.mode, test.want, got)
			}
		})
	}
}

func TestImageEnhancerPreservesDetailedAndComposerPrompts(t *testing.T) {
	got := System("t2i", false)
	for _, want := range []string{"prompt composer", "do not elaborate it further", "Preserve visible text exactly"} {
		if !strings.Contains(got, want) {
			t.Fatalf("image enhancer rules do not contain %q", want)
		}
	}
}
