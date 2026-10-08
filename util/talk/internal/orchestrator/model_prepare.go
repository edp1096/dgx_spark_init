package orchestrator

import "context"

type preparationTokenKey struct{}

func (c *Controller) prepareSingleService(ctx context.Context, component Component, action, token string) error {
	if action == "model" && len(componentModelAssets(component)) > 0 {
		return c.ensureModelAssets(ctx, component, token)
	}
	return c.prepareOrStartComponent(context.WithValue(ctx, preparationTokenKey{}, token), component, true)
}
