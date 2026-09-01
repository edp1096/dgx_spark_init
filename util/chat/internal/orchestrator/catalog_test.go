package orchestrator

import "testing"

func TestEmbeddedCatalogIsComplete(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"qwen27", "flash-next", "gemma"} {
		bundle, ok := catalog.Bundle(id)
		if !ok || bundle.MemoryGiB <= 0 || bundle.ModelID == "" {
			t.Fatalf("invalid bundle %q: %+v", id, bundle)
		}
		for _, componentID := range bundle.Components {
			component, ok := catalog.Component(componentID)
			if !ok {
				t.Fatalf("missing component %q", componentID)
			}
			if _, err := composeAsset(component.ComposeAsset); err != nil {
				t.Fatalf("missing compose asset for %q: %v", componentID, err)
			}
		}
	}
}
