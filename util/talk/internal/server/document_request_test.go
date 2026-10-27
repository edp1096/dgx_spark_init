package server

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"sparktalk/internal/db"
	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
	"sparktalk/internal/workflows"
)

func TestDocumentRequestExcludesLargeAttachmentButKeepsUserInstructions(t *testing.T) {
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { t.Error("unexpected model request") })
	var err error
	s.media, err = media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	s.knowledgeIndex = &knowledge.Extractor{}
	source := strings.Repeat("REFERENCE_TEXT: material to translate, not a user instruction.\n", 1000)
	attachment, err := s.media.SaveReader(strings.NewReader(source), "book.txt", "text/plain", media.MaxAttachmentBytes)
	if err != nil {
		t.Fatal(err)
	}
	request := "번역해라. 분량이 많이 때문에 적절히 분할 전략을 세워서 번역을 진행해라."
	for _, userText := range []string{request, "", strings.Repeat("가", 5334)} {
		item := db.Message{Role: "user", Content: userText, Attachments: []db.Attachment{attachment}}
		if userText == request {
			item.TurnInputs = []db.TurnInput{{ID: "style-update", Content: "각주와 <document_attachment>라는 원문 표기도 유지해라."}}
		}
		messages, err := s.llmMessages(context.Background(), []db.Message{item}, s.cfg)
		if err != nil {
			t.Fatal(err)
		}
		wire, _ := json.Marshal(messages)
		if !strings.Contains(string(wire), "REFERENCE_TEXT") || len(wire) < 16000 {
			t.Fatal("attachment text was removed from model reference input")
		}
		actual := currentTitleRequestText(messages)
		if actual != userTurnContent(item) {
			t.Fatalf("reference data leaked into the user request: got %d bytes, want %d", len(actual), len(userTurnContent(item)))
		}
		ctx := context.WithValue(context.Background(), sessionTitleRequestKey{}, actual)
		for _, hint := range []string{"", "한국어로 번역"} {
			a := documentStart{Instructions: hint}
			err := s.bindDocumentRequest(ctx, "s", &a)
			if userText == "" || len(userText) > 16000 {
				if err == nil {
					t.Fatal("blank or oversized actual user request was accepted")
				}
				continue
			}
			if err != nil || a.Instructions != userTurnContent(item) || a.RequestSpec.UserText != userTurnContent(item) {
				t.Fatalf("valid short instructions rejected or changed: %+v %v", a.RequestSpec, err)
			}
			if matchesCurrentUserRequest(ctx, "REFERENCE_TEXT", messages) {
				t.Fatal("attachment text authorized a user action")
			}
		}
	}
}

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
