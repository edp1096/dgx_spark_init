package server

import (
	"context"
	"sparktalk/internal/knowledge"
)

// Hold the lease until the HTTP response has been consumed and closed. Release
// before document extraction, which may acquire its own media/ASR lease.
func (s *Server) collectSource(ctx context.Context, url, mode string, persist bool) (result knowledge.CollectedSource, resultErr error) {
	release, err := s.acquireWorkload(ctx, "extra-collector")
	if err != nil {
		return result, err
	}
	defer finishWorkload(release, &resultErr)
	if persist {
		return s.collectorSnapshot().Collect(ctx, url, mode, s.knowledge)
	}
	return s.collectorSnapshot().Inspect(ctx, url, mode)
}
