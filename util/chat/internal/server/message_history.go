package server

import (
	"fmt"
	"regexp"
	"strings"

	"sparktalk/internal/db"
)

var embeddedDataPattern = regexp.MustCompile(`(?i)data:[^;\s]+;base64,[^\s"']+`)

// modelHistory separates the visible transcript from the model-facing
// transcript. Failed and cancelled requests remain visible, but are converted
// to short text-only exchanges so broken media and large error payloads cannot
// poison later requests. currentPendingID is the one request allowed to carry
// its original attachments while it is being generated.
func modelHistory(items []db.Message, currentPendingID int64) []db.Message {
	out := make([]db.Message, 0, len(items))
	for _, item := range items {
		status := item.Status
		if status == "" {
			status = db.MessageCompleted
		}
		switch status {
		case db.MessageCompleted:
			out = append(out, item)
		case db.MessagePending:
			if item.ID == currentPendingID {
				out = append(out, item)
			}
		case db.MessageFailed, db.MessageCancelled:
			if item.Role != "user" {
				continue
			}
			label := "실패한 이전 요청"
			fallback := "모델 요청이 완료되지 않았습니다."
			if status == db.MessageCancelled {
				label = "사용자가 중지한 이전 요청"
				fallback = "사용자가 생성을 중지했습니다."
			}
			request := "[" + label + "]\n요청: " + compactHistoryText(item.Content, 800)
			for _, attachment := range item.Attachments {
				request += fmt.Sprintf("\n첨부: %s (%s, %s)", compactHistoryText(attachment.Name, 160), attachment.MIME, humanBytes(attachment.Size))
			}
			failure := compactHistoryText(item.Error, 800)
			if failure == "" {
				failure = fallback
			}
			out = append(out,
				db.Message{ID: item.ID, SessionID: item.SessionID, Role: "user", Status: db.MessageCompleted, Content: request},
				db.Message{ID: item.ID, SessionID: item.SessionID, Role: "assistant", Status: db.MessageCompleted, Content: "[처리 결과] " + failure},
			)
		}
	}
	return out
}

func compactHistoryText(value string, limit int) string {
	value = embeddedDataPattern.ReplaceAllString(value, "[첨부 데이터 생략]")
	value = strings.Join(strings.Fields(value), " ")
	runes := []rune(value)
	if len(runes) > limit {
		return string(runes[:limit]) + "…"
	}
	return value
}

func humanBytes(size int64) string {
	const mb = 1024 * 1024
	if size >= mb {
		return fmt.Sprintf("%.1f MB", float64(size)/mb)
	}
	if size >= 1024 {
		return fmt.Sprintf("%.1f KB", float64(size)/1024)
	}
	return fmt.Sprintf("%d B", size)
}
