package server

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

func TestDocumentUserRequestOverridesInventedToolConstraint(t *testing.T) {
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { t.Error("unexpected model request") })
	ctx := context.WithValue(context.Background(), sessionTitleRequestKey{}, "이 문서 전체를 번역해라.")
	a := documentStart{Instructions: "고유명사는 UNKNOWN_NAME으로 고정하라.", Glossary: map[string]string{"Name": "UNKNOWN_NAME"}}
	if err := s.bindDocumentRequest(ctx, "s", &a); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(a.Instructions, "UNKNOWN_NAME") || len(a.Glossary) != 0 || a.RequestSpec.Hash != documentTextHash(a.Instructions) || !strings.Contains(a.RequestSpec.ToolHints, "UNKNOWN_NAME") {
		t.Fatalf("invented instruction acquired authority: %+v", a)
	}
	prior, err := s.db.AddMessage("s", "user", "앞서 지정한 문체는 존댓말이다.", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	assistant, err := s.db.AddMessage("s", "assistant", "UNKNOWN_NAME으로 정했습니다.", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	a.InstructionMessageIDs = []int64{prior.ID}
	if err = s.bindDocumentRequest(ctx, "s", &a); err != nil || !strings.Contains(a.Instructions, "존댓말") {
		t.Fatal("lost explicit prior user instruction", err)
	}
	a.InstructionMessageIDs = []int64{assistant.ID}
	if err = s.bindDocumentRequest(ctx, "s", &a); err == nil {
		t.Fatal("assistant constraint promoted to user instruction")
	}
}

func TestDocumentRepairRetainsActualInstructionsAndContext(t *testing.T) {
	_, backend := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		messages := body["messages"].([]any)
		text := messages[len(messages)-1].(map[string]any)["content"].(string)
		for _, want := range []string{"ACTUAL_REQUEST", "SOURCE_CONTEXT", "ACCEPTED_CONTEXT"} {
			if !strings.Contains(text, want) {
				t.Error("missing repair context", want)
			}
		}
		if strings.Contains(text, "INVENTED_CONSTRAINT") {
			t.Error("invented instruction in model request")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"entries\\\":{\\\"p000001\\\":\\\"번역\\\"}}\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	})
	run := workflows.Run{Document: &workflows.DocumentTask{Kind: "translate", Model: "model", Effort: "off", Instructions: "INVENTED_CONSTRAINT", RequestSpec: &workflows.DocumentRequestSpec{UserText: "ACTUAL_REQUEST"}, SourceContext: "SOURCE_CONTEXT", ContextTail: "ACCEPTED_CONTEXT"}}
	u := workflows.DocumentUnit{Stage: "repair", Inputs: []workflows.DocumentInput{{ID: "s:b:0", Text: "Source"}}}
	if _, err := requestDocumentUnit(context.Background(), llm.New(backend.URL, "model", ""), run, u, 8192); err != nil {
		t.Fatal(err)
	}
}

func TestDocumentSourceContextExcludesTranslatedRangeAndRespectsBudget(t *testing.T) {
	var units []workflows.DocumentUnit
	for i, text := range []string{"Near first.", "Prior paragraph.", "TARGET TEXT", "Next paragraph.", "Last neighbour."} {
		id := documentLocalID(i)
		units = append(units, workflows.DocumentUnit{Ordinal: i, Inputs: []workflows.DocumentInput{{ID: id, Source: "s", Block: id, Text: text}}})
	}
	actual := documentSourceContext(units, units[2], 2048)
	if strings.Contains(actual, "TARGET TEXT") || !strings.Contains(actual, "Prior paragraph.") || !strings.Contains(actual, "Next paragraph.") {
		t.Fatal(actual)
	}
	if got := documentSourceContext(units, units[2], 1); got != "" {
		t.Fatal("context exceeded budget", got)
	}
}
