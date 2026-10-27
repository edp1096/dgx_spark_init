package server

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestMusicToolOnlyOffersACEAndRejectsRemovedModelBeforeWork(t *testing.T) {
	s := &Server{}
	if _, err := s.generateMusic(context.Background(), musicRequest{Model: "heartmula", Prompt: "solo piano", Duration: 30}); err == nil {
		t.Fatal("removed model accepted")
	}
	reg := completionToolRegistry{}
	reg.handlers = map[string]registeredToolHandler{}
	s.registerMusicTool(&reg)
	for _, tool := range reg.definitions {
		if tool.Function.Name != "music_generate" {
			continue
		}
		var schema struct {
			Properties map[string]struct {
				Enum []string `json:"enum"`
			} `json:"properties"`
		}
		if err := json.Unmarshal(tool.Function.Parameters, &schema); err != nil {
			t.Fatal(err)
		}
		if got := strings.Join(schema.Properties["model"].Enum, ","); got != "ace-turbo,ace-xl" {
			t.Fatal(got)
		}
		return
	}
	t.Fatal("missing music tool")
}
