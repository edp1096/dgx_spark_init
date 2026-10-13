package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"sparktalk/internal/workflows"
)

func documentTextHash(text string) string {
	h := sha256.Sum256([]byte(text))
	return hex.EncodeToString(h[:])
}

func (s *Server) bindDocumentRequest(ctx context.Context, session string, a *documentStart) error {
	if len(a.Instructions) > 16000 {
		return fmt.Errorf("document task instructions exceed 16000 UTF-8 bytes")
	}
	request, bound := ctx.Value(sessionTitleRequestKey{}).(string)
	if !bound || strings.TrimSpace(request) == "" {
		return fmt.Errorf("document task requires the actual user request")
	}
	if len(a.InstructionMessageIDs) > 8 {
		return fmt.Errorf("at most eight earlier user instruction messages may be referenced")
	}
	var anchor int64
	if turn := turnFrom(ctx); turn != nil {
		turn.mu.Lock()
		anchor = turn.anchor
		turn.mu.Unlock()
	}
	spec := &workflows.DocumentRequestSpec{Origin: "user_turn", ToolHints: a.Instructions}
	var earlier []string
	if len(a.InstructionMessageIDs) > 0 {
		messages, err := s.db.Messages(session)
		if err != nil {
			return err
		}
		seen := map[int64]bool{}
		for _, id := range a.InstructionMessageIDs {
			if seen[id] {
				continue
			}
			seen[id] = true
			found := false
			for _, message := range messages {
				if message.ID != id {
					continue
				}
				if message.Role != "user" || anchor > 0 && id > anchor {
					return fmt.Errorf("instruction reference %d is not an eligible user message", id)
				}
				if id != anchor {
					earlier = append(earlier, fmt.Sprintf("Earlier user message %d:\n%s", id, userTurnContent(message)))
					spec.MessageIDs = append(spec.MessageIDs, id)
				}
				found = true
				break
			}
			if !found {
				return fmt.Errorf("user instruction message %d was not found in this conversation", id)
			}
		}
	}
	if anchor > 0 {
		spec.MessageIDs = append(spec.MessageIDs, anchor)
	}
	if len(earlier) > 0 {
		request = strings.Join(earlier, "\n\n") + "\n\nCurrent user request (takes precedence):\n" + request
	}
	if len(request) > 16000 {
		return fmt.Errorf("explicit document instructions exceed 16000 UTF-8 bytes")
	}
	spec.UserText, spec.Hash = request, documentTextHash(request)
	a.RequestSpec = spec
	a.Instructions = request
	// Tool-inferred pairs are not user-pinned. Explicit mappings remain in the
	// exact user request above; direct UI/API requests may still pin a glossary.
	a.Glossary = nil
	return nil
}

func documentUserInstructions(task *workflows.DocumentTask) string {
	if task.RequestSpec != nil {
		return task.RequestSpec.UserText
	}
	return task.Instructions // legacy jobs and direct UI requests
}
