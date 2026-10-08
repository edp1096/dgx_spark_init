package orchestrator

import "sparktalk/internal/modelidentity"

// Normalize persisted references before indexing or importing a catalog.
func NormalizeRuntimeIdentities(catalog *Catalog) {
	for i := range catalog.Components {
		x := &catalog.Components[i]
		x.ID = modelidentity.CanonicalID(x.ID)
		x.Model = modelidentity.CanonicalID(x.Model)
		x.Container = modelidentity.CanonicalContainer(x.Container)
		x.ComposeAsset = modelidentity.CanonicalCompose(x.ComposeAsset)
	}
	for i := range catalog.Bundles {
		b := &catalog.Bundles[i]
		b.ID = modelidentity.CanonicalID(b.ID)
		b.ModelID = modelidentity.CanonicalID(b.ModelID)
		b.ModelType = modelidentity.CanonicalID(b.ModelType)
		for j := range b.Components {
			b.Components[j] = modelidentity.CanonicalID(b.Components[j])
		}
		bindings := make(map[string]Deployment, len(b.Bindings))
		for id, binding := range b.Bindings {
			if binding.Model != nil {
				value := modelidentity.CanonicalID(*binding.Model)
				binding.Model = &value
			}
			if binding.Container != nil {
				value := modelidentity.CanonicalContainer(*binding.Container)
				binding.Container = &value
			}
			bindings[modelidentity.CanonicalID(id)] = binding
		}
		if b.Bindings != nil {
			b.Bindings = bindings
		}
	}
}
