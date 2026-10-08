package config

import (
	"sparktalk/internal/modelidentity"
	"sparktalk/internal/orchestrator"
)

func (c *Config) migrateQwen38FNEXL3Identity() {
	c.Runtime.Bundle = modelidentity.CanonicalID(c.Runtime.Bundle)
	c.Runtime.ActiveBundle = modelidentity.CanonicalID(c.Runtime.ActiveBundle)
	c.Model.DefaultModel = modelidentity.CanonicalID(c.Model.DefaultModel)
	c.Model.ModelType = modelidentity.CanonicalID(c.Model.ModelType)
	if c.Runtime.Catalog != nil {
		orchestrator.NormalizeRuntimeIdentities(c.Runtime.Catalog)
	}
}
