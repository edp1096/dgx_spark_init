package llm

import (
	"context"
	"encoding/json"
)

type structuredOutputKey struct{}
type structuredOutput struct {
	Name   string
	Schema json.RawMessage
}

// WithJSONSchema is scoped to one request. It does not alter chat settings or
// claim support on backends which reject structured output.
func WithJSONSchema(ctx context.Context, name string, schema json.RawMessage) context.Context {
	return context.WithValue(ctx, structuredOutputKey{}, structuredOutput{Name: name, Schema: append(json.RawMessage(nil), schema...)})
}
