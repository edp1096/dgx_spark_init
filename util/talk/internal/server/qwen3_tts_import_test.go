package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"sparktalk/internal/orchestrator"
)

func TestCatalogImportMigratesRetiredTTSBeforeRecipeValidation(t *testing.T) {
	cat, err := orchestrator.LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for i := range cat.Components {
		x := &cat.Components[i]
		x.ModelPresentation = nil
		if x.ID == "qwen3-tts" {
			x.ID, x.Model, x.ComposeAsset = "magpie-tts", "magpietts", "compose.magpie-tts.yaml"
			x.Container = "sparktalk-magpie-tts"
		}
	}
	for i := range cat.Bundles {
		b := &cat.Bundles[i]
		for j, id := range b.Components {
			if id == "qwen3-tts" {
				b.Components[j] = "magpie-tts"
			}
		}
		if v, ok := b.Bindings["qwen3-tts"]; ok {
			b.Bindings["magpie-tts"] = v
			delete(b.Bindings, "qwen3-tts")
		}
	}
	raw, err := json.Marshal(cat)
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{}
	w := httptest.NewRecorder()
	s.runtimeCatalogParse(w, httptest.NewRequest(http.MethodPost, "/api/runtime/catalog/parse", bytes.NewReader(raw)))
	if w.Code != http.StatusOK {
		t.Fatalf("legacy catalog rejected: %d %s", w.Code, w.Body.String())
	}
	var migrated orchestrator.Catalog
	if err = json.Unmarshal(w.Body.Bytes(), &migrated); err != nil {
		t.Fatal(err)
	}
	for _, x := range migrated.Components {
		if x.ID == "magpie-tts" || x.ComposeAsset == "compose.magpie-tts.yaml" {
			t.Fatal("retired recipe returned", x)
		}
	}
	if s.runtime != nil {
		t.Fatal("import started runtime controller")
	}
}
