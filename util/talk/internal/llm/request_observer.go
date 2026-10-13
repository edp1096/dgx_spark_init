package llm

import "context"

type requestObserverKey struct{}

// WithRequestObserver observes the serialized request body, excluding transport
// credentials. The observer receives a copy and cannot alter the sent payload.
func WithRequestObserver(ctx context.Context, observer func([]byte)) context.Context {
	return context.WithValue(ctx, requestObserverKey{}, observer)
}
