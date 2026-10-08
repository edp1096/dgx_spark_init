package server

import (
	"context"
	"sort"
	"strings"
	"sync"

	"sparktalk/internal/db"
)

type attachmentRequestKey struct{}

type conversationAttachment struct {
	Attachment db.Attachment
	Origin     string
	MessageID  int64
	Index      int
}

// The selected retry/edit branch is immutable during completion. Tool outputs
// are added separately so every stage can use them before the response commits.
type attachmentRequest struct {
	mu        sync.Mutex
	sessionID string
	inputs    []conversationAttachment
	outputs   []conversationAttachment
}

func withRequestAttachments(ctx context.Context, sessionID string, messages []db.Message) context.Context {
	request := &attachmentRequest{sessionID: sessionID}
	for _, message := range messages {
		for index, attachment := range message.Attachments {
			request.inputs = append(request.inputs, conversationAttachment{attachment, message.Role, message.ID, index + 1})
		}
	}
	return context.WithValue(ctx, attachmentRequestKey{}, request)
}

func recordRequestAttachments(ctx context.Context, sessionID, origin string, attachments []db.Attachment) {
	request, ok := ctx.Value(attachmentRequestKey{}).(*attachmentRequest)
	if !ok || request.sessionID != sessionID {
		return
	}
	request.mu.Lock()
	defer request.mu.Unlock()
	for _, attachment := range attachments {
		request.outputs = append(request.outputs, conversationAttachment{attachment, origin, 0, len(request.outputs) + 1})
	}
}

func (s *Server) conversationAttachments(ctx context.Context, sessionID string) ([]conversationAttachment, error) {
	var candidates []conversationAttachment
	if request, ok := ctx.Value(attachmentRequestKey{}).(*attachmentRequest); ok && request.sessionID == sessionID {
		request.mu.Lock()
		candidates = append(candidates, request.inputs...)
		candidates = append(candidates, request.outputs...)
		request.mu.Unlock()
	} else {
		messages, err := s.db.Messages(sessionID)
		if err != nil {
			return nil, err
		}
		for _, message := range messages {
			for index, attachment := range message.Attachments {
				candidates = append(candidates, conversationAttachment{attachment, message.Role, message.ID, index + 1})
			}
		}
	}
	// Direct image tool execution and resumed workflows also retain generated
	// images here. Include them even when they have not passed through a registry.
	if images, ok := ctx.Value(turnImageKey{}).(*turnImages); ok && images.sessionID == sessionID {
		images.mu.Lock()
		ids := make([]string, 0, len(images.items))
		for id := range images.items {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		for index, id := range ids {
			candidates = append(candidates, conversationAttachment{images.items[id], "generated_this_turn", 0, index + 1})
		}
		images.mu.Unlock()
	}
	seen := make(map[string]bool)
	items := make([]conversationAttachment, 0, len(candidates))
	for _, item := range candidates {
		if item.Attachment.ID != "" && !seen[item.Attachment.ID] {
			items = append(items, item)
			seen[item.Attachment.ID] = true
		}
	}
	return items, nil
}

func (s *Server) sessionImageAttachmentsForContext(ctx context.Context, sessionID string) (map[string]db.Attachment, error) {
	attachments, err := s.conversationAttachments(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	items := make(map[string]db.Attachment)
	for _, item := range attachments {
		if strings.HasPrefix(item.Attachment.MIME, "image/") {
			items[item.Attachment.ID] = item.Attachment
		}
	}
	return items, nil
}
