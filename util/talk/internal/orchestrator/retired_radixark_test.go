package orchestrator

import (
	"encoding/json"
	"os"
	"testing"
)

// Historical lifecycle coverage uses an explicit fixture, never a selectable built-in.
func legacyRadixArkCatalog(t *testing.T) Catalog {
	t.Helper()
	cat, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("testdata/retired-radixark.json")
	if err != nil {
		t.Fatal(err)
	}
	var old Catalog
	if err := json.Unmarshal(raw, &old); err != nil {
		t.Fatal(err)
	}
	cat.Components = append(cat.Components, old.Components...)
	cat.Bundles = append(cat.Bundles, old.Bundles...)
	cat, err = ValidateCatalog(cat)
	if err != nil {
		t.Fatal(err)
	}
	return cat
}
