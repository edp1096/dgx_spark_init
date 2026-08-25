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
		{mode: "edit", want: "one short, direct English edit sentence"},
		{mode: "edit_control", want: "application itself adds"},
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

func TestIdentityEditModuleContextIsSystemOwnedAndSpecific(t *testing.T) {
	got := EditModuleContext("tryon", []string{"identity", "background"})
	for _, want := range []string{"supporting image supplies the complete replacement outfit", "identity, background", "do not recite"} {
		if !strings.Contains(got, want) {
			t.Fatalf("edit module context does not contain %q: %s", want, got)
		}
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
