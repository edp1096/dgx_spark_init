package server

import (
	"testing"

	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

func TestDocumentEntitiesRequireSourceEvidenceAndKeepAnimalsAndWorks(t *testing.T) {
	u := workflows.DocumentUnit{Inputs: []workflows.DocumentInput{{ID: "s:b1:0", Source: "s", Block: "b1", Group: "g1", Plain: true, Text: "Pip was a dog. She read The River."}}}
	result := llm.StreamResult{Content: `{"entries":{},"terms":[{"source":"Pip","target":"핍","kind":"animal","evidence":{"id":"p000001","quote":"Pip was a dog."}},{"source":"The River","target":"강","kind":"work","evidence":{"id":"p000001","quote":"She read The River."}},{"source":"Invented","target":"꾸며낸 이름","kind":"person","evidence":{"id":"p000001","quote":"Pip was a dog."}},{"source":"Pip","target":"가짜","kind":"animal","evidence":{"id":"missing","quote":"Pip was a dog."}}]}`}
	entities := documentEntityCandidates(u, result, nil)
	if len(entities) != 2 || entities[0].Kind != "animal" || entities[1].Kind != "work" || entities[0].Status != "candidate" {
		t.Fatal(entities)
	}
	if len(documentEntityCandidates(u, result, entities)) != 0 {
		t.Fatal("unchanged proposal created a new version")
	}
	selected, versions := documentRelevantEntities(workflows.DocumentUnit{Inputs: []workflows.DocumentInput{{Text: "Pip returned."}}}, entities, 1024)
	if len(selected) != 0 {
		t.Fatal("unreviewed proposal propagated into later translation")
	}
	for i := range entities {
		entities[i].Status = "reviewed"
	}
	selected, versions = documentRelevantEntities(workflows.DocumentUnit{Inputs: []workflows.DocumentInput{{Text: "Pip returned."}}}, entities, 1024)
	if len(selected) != 1 || len(versions) != 1 {
		t.Fatal("unrelated records injected", selected)
	}
	selected, _ = documentRelevantEntities(u, entities, 1)
	if len(selected) != 0 {
		t.Fatal("entity context exceeded budget")
	}
}

func TestDocumentEntityDependencyReviewDoesNotCreateVersionCycle(t *testing.T) {
	entity := workflows.DocumentEntity{ID: "pip", Version: 2, Source: "Pip", Target: "핍", Kind: "animal", Status: "reviewed", Origin: "model_candidate", Evidence: []workflows.DocumentEntityEvidence{{InputID: "source:a"}}}
	units := []workflows.DocumentUnit{{Status: "completed", Inputs: []workflows.DocumentInput{{ID: "source:a", Text: "Pip ran."}}, Output: map[string]string{"source:a": "핍이 달렸다."}, Quality: &workflows.DocumentUnitQuality{Status: "pending", EntityVersions: map[string]int{"other": 1}}}}
	if updates := documentEntityReviewUpdates([]workflows.DocumentEntity{entity}, units); len(updates) != 0 {
		t.Fatal("dependency invalidation spuriously downgraded an unchanged referent", updates)
	}
}

func TestDocumentEntityMentionsDoNotTurnNamePrefixesIntoIdentities(t *testing.T) {
	for _, tc := range []struct {
		text, source string
		want         bool
	}{{"Pippin came home.", "Pip", false}, {"Pip's footprints.", "Pip", true}, {"Annex A", "Ann", false}, {"Pipは帰った。", "Pip", true}, {"カアの足音", "カア", true}} {
		if got := documentEntityMention(tc.text, tc.source); got != tc.want {
			t.Fatal(tc, got)
		}
	}
}
