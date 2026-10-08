package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
)

func emitAttachmentTestCompletion(w http.ResponseWriter, call *llm.ToolCall) {
	w.Header().Set("Content-Type", "text/event-stream")
	delta := map[string]any{"content": "finished"}
	finish := "stop"
	if call != nil {
		delta = map[string]any{"tool_calls": []any{map[string]any{"index": 0, "id": call.ID, "type": "function", "function": call.Function}}}
		finish = "tool_calls"
	}
	chunk, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": delta, "finish_reason": finish}}})
	fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", chunk)
}

func serveAttachmentTestPDF(w http.ResponseWriter) {
	_ = json.NewEncoder(w).Encode(map[string]any{"files": []map[string]string{{"name": "document.pdf", "mime": "application/pdf", "data": base64.StdEncoding.EncodeToString([]byte("%PDF-1.4\nfixture"))}}})
}

func TestHTTPSelectedBranchAttachmentsReachTools(t *testing.T) {
	for _, scenario := range []string{"older_variant", "new_attachment_during_edit"} {
		for _, tool := range []string{"attachment_read", "image_generate", "document_generate"} {
			t.Run(scenario+"/"+tool, func(t *testing.T) {
				s, original := testImageServer(t)
				t.Cleanup(func() { _ = s.tasks.Wait(context.Background()) })
				messages, err := s.db.Messages("session")
				if err != nil {
					t.Fatal(err)
				}
				parent := messages[0]
				answer, err := s.db.AddMessage("session", "assistant", "old answer", "", nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				var replacementBytes bytes.Buffer
				portrait := image.NewRGBA(image.Rect(0, 0, 12, 20))
				portrait.SetRGBA(0, 0, color.RGBA{B: 255, A: 255})
				if err := png.Encode(&replacementBytes, portrait); err != nil {
					t.Fatal(err)
				}
				replacement, err := s.media.SaveReader(bytes.NewReader(replacementBytes.Bytes()), "replacement.png", "image/png", media.MaxImageBytes)
				if err != nil {
					t.Fatal(err)
				}
				selected, excluded := original, replacement
				request := map[string]any{"tools_enabled": true, "user_variant": 0}
				path := fmt.Sprintf("/api/messages/%d/retry", answer.ID)
				if scenario == "older_variant" {
					if err := s.db.AppendEditedBranch(parent.ID, "use replacement", []db.Attachment{replacement}, "new answer", "", nil); err != nil {
						t.Fatal(err)
					}
				} else {
					selected, excluded = replacement, original
					request = map[string]any{"tools_enabled": true, "content": "use the new image", "attachments": []db.Attachment{replacement}}
					path = fmt.Sprintf("/api/messages/%d/edit", parent.ID)
				}
				selectedData, err := s.media.DataURL(selected)
				if err != nil {
					t.Fatal(err)
				}
				selectedBytes := onePixelPNG
				if selected.ID == replacement.ID {
					selectedBytes = replacementBytes.Bytes()
				}
				workerCalls := 0
				worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					workerCalls++
					if tool == "image_generate" {
						if err := r.ParseMultipartForm(1 << 20); err != nil {
							t.Error(err)
							return
						}
						defer r.MultipartForm.RemoveAll()
						files := r.MultipartForm.File["image"]
						if len(files) != 1 {
							t.Errorf("expected selected photo, got %d", len(files))
							return
						}
						file, err := files[0].Open()
						if err != nil {
							t.Error(err)
							return
						}
						got, _ := io.ReadAll(file)
						file.Close()
						if !bytes.Equal(got, selectedBytes) {
							t.Error("engine received a different photo")
						}
						_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
					} else {
						var body struct {
							Sections []struct {
								Images []struct {
									Data string `json:"data"`
								} `json:"images"`
							} `json:"sections"`
						}
						if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
							t.Error(err)
							return
						}
						if len(body.Sections) != 1 || len(body.Sections[0].Images) != 1 {
							t.Error("document image missing")
							return
						}
						encoded, err := base64.StdEncoding.DecodeString(body.Sections[0].Images[0].Data)
						if err != nil {
							t.Error(err)
							return
						}
						got, err := png.Decode(bytes.NewReader(encoded))
						if err != nil {
							t.Error(err)
							return
						}
						want, _ := png.Decode(bytes.NewReader(selectedBytes))
						if got.Bounds() != want.Bounds() {
							t.Error("document received the wrong photo dimensions")
						}
						gr, gg, gb, ga := got.At(0, 0).RGBA()
						wr, wg, wb, wa := want.At(0, 0).RGBA()
						if gr != wr || gg != wg || gb != wb || ga != wa {
							t.Error("document received a different photo")
						}
						serveAttachmentTestPDF(w)
					}
				}))
				defer worker.Close()
				args := map[string]any{"attachment_id": selected.ID}
				if tool == "image_generate" {
					args = map[string]any{"operation": "reference_generate", "prompt": "use this reference", "reference_images": []imageReference{{selected.ID, "selected original"}}, "head_targets": []imageHeadTarget{}}
				}
				if tool == "document_generate" {
					args = map[string]any{"format": "pdf", "title": "selected photo", "sections": []map[string]any{{"paragraphs": []string{"photo"}, "images": []map[string]string{{"image_id": selected.ID}}}}}
				}
				raw, _ := json.Marshal(args)
				calls := 0
				backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.Method == http.MethodGet {
						_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]any{{"id": "model", "max_model_len": 32768}}})
						return
					}
					var body struct {
						Messages []llm.Message `json:"messages"`
						Stream   bool          `json:"stream"`
					}
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
						return
					}
					if !body.Stream {
						_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": "edited"}}}})
						return
					}
					calls++
					if calls == 1 {
						payload, _ := json.Marshal(body.Messages)
						if !strings.Contains(string(payload), selected.ID) || !strings.Contains(string(payload), selectedData) || strings.Contains(string(payload), excluded.ID) {
							t.Error("model input/catalog does not match selected request")
						}
						emitAttachmentTestCompletion(w, &llm.ToolCall{ID: "selected", Function: llm.FunctionCall{Name: tool, Arguments: string(raw)}})
					} else {
						if tool == "attachment_read" {
							payload, _ := json.Marshal(body.Messages)
							if !strings.Contains(string(payload), selectedData) || strings.Contains(string(payload), `origin=\"assistant\"`) {
								t.Error("uploaded photo was lost or relabeled as generated")
							}
						}
						emitAttachmentTestCompletion(w, nil)
					}
				}))
				defer backend.Close()
				s.cfg = config.Config{Model: config.ModelConfig{Endpoint: backend.URL, DefaultModel: "model"}, Context: config.ContextConfig{WindowTokens: 32768}, Tools: config.ToolsConfig{Enabled: true, MaxRounds: 3}, Image: config.ImageConfig{Enabled: true, Endpoint: worker.URL, Model: "test", Mode: "qwen-image21", DefaultSize: "512x512", Timeout: "2s"}}
				s.cfg.Extra.DocumentsEnabled, s.cfg.Extra.DocumentsEndpoint = true, worker.URL
				s.llm = llm.New(backend.URL, "model", "")
				body, _ := json.Marshal(request)
				w := httptest.NewRecorder()
				s.messageAction(w, httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body)))
				if w.Code != http.StatusOK || strings.Contains(w.Body.String(), "event: error") || !strings.Contains(w.Body.String(), "event: done") {
					t.Fatalf("HTTP request failed: %s", w.Body.String())
				}
				stored, err := s.db.Messages("session")
				if err != nil {
					t.Fatal(err)
				}
				trace := stored[len(stored)-1].ToolTrace
				if len(trace) != 1 || trace[0].Name != tool || trace[0].Error != "" {
					t.Fatalf("selected attachment tool failed: %+v", trace)
				}
				if tool == "image_generate" && !strings.Contains(trace[0].Result, `"origin":"user"`) {
					t.Fatal("selected upload provenance lost")
				}
				wantCalls := 1
				if tool == "attachment_read" {
					wantCalls = 0
				}
				if workerCalls != wantCalls {
					t.Fatalf("unexpected worker calls: %d", workerCalls)
				}
			})
		}
	}
}

func TestImageThenDocumentInSameCompletion(t *testing.T) {
	s, _ := testImageServer(t)
	if _, err := s.db.AddMessage("session", "user", "참조 없이 그림을 만들고 문서에 넣어라", "", nil, nil); err != nil {
		t.Fatal(err)
	}
	history, err := s.db.Messages("session")
	if err != nil {
		t.Fatal(err)
	}
	documentCalls := 0
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/images/generations" {
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
			return
		}
		documentCalls++
		var body struct {
			Sections []struct {
				Images []struct {
					Data string `json:"data"`
				} `json:"images"`
			} `json:"sections"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if len(body.Sections) != 1 || len(body.Sections[0].Images) != 1 || body.Sections[0].Images[0].Data == "" {
			t.Error("generated image missing from document")
			return
		}
		serveAttachmentTestPDF(w)
	}))
	defer worker.Close()
	calls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]any{{"id": "model", "max_model_len": 32768}}})
			return
		}
		calls++
		var body struct {
			Messages []llm.Message `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		switch calls {
		case 1:
			emitAttachmentTestCompletion(w, &llm.ToolCall{ID: "image", Function: llm.FunctionCall{Name: "image_generate", Arguments: `{"operation":"generate","prompt":"diagram"}`}})
		case 2:
			var generatedID string
			for _, message := range body.Messages {
				if message.ToolCallID == "image" {
					var result struct {
						Attachments []db.Attachment `json:"attachments"`
					}
					if err := json.Unmarshal([]byte(fmt.Sprint(message.Content)), &result); err != nil || len(result.Attachments) != 1 {
						t.Errorf("image result missing: %v", err)
						return
					}
					generatedID = result.Attachments[0].ID
				}
			}
			if generatedID == "" {
				t.Error("no generated image ID")
				return
			}
			stored, _ := s.sessionImageAttachments("session")
			if _, exists := stored[generatedID]; exists {
				t.Error("fixture must embed image before database commit")
			}
			args := fmt.Sprintf(`{"format":"pdf","title":"diagram","sections":[{"paragraphs":["diagram"],"images":[{"image_id":%q}]}]}`, generatedID)
			emitAttachmentTestCompletion(w, &llm.ToolCall{ID: "document", Function: llm.FunctionCall{Name: "document_generate", Arguments: args}})
		default:
			emitAttachmentTestCompletion(w, nil)
		}
	}))
	defer backend.Close()
	cfg := config.Config{Context: config.ContextConfig{WindowTokens: 32768}, Tools: config.ToolsConfig{Enabled: true, MaxRounds: 4}, Image: config.ImageConfig{Enabled: true, Endpoint: worker.URL, Model: "test", Mode: "qwen-image21", DefaultSize: "512x512", Timeout: "2s"}}
	cfg.Extra.DocumentsEnabled, cfg.Extra.DocumentsEndpoint = true, worker.URL
	s.cfg = cfg
	result, err := s.runContextCompletion(context.Background(), "session", history, "model", "none", cfg, llm.New(backend.URL, "model", ""), true, func(string, any) error { return nil }, func(db.Attachment) error { return nil })
	if err != nil || documentCalls != 1 || len(result.ToolTrace) != 2 || len(result.Attachments) != 2 {
		t.Fatalf("image-to-document chain failed: docs=%d result=%+v err=%v", documentCalls, result, err)
	}
	for _, event := range result.ToolTrace {
		if event.Error != "" {
			t.Fatalf("chain tool failed: %+v", event)
		}
	}
}

func TestRequestAttachmentsKeepSessionIsolationAndIgnoreFailedOutputs(t *testing.T) {
	s, original := testImageServer(t)
	if _, err := s.db.CreateSession("other", "other", "model", "none"); err != nil {
		t.Fatal(err)
	}
	ctx := withRequestAttachments(context.Background(), "session", []db.Message{{Role: "user", Attachments: []db.Attachment{original}}})
	items, err := s.sessionImageAttachmentsForContext(ctx, "other")
	if err != nil || len(items) != 0 {
		t.Fatalf("request attachment crossed sessions: %v %v", items, err)
	}
	output, err := s.media.SaveReader(bytes.NewReader(onePixelPNG), "output.png", "image/png", media.MaxImageBytes)
	if err != nil {
		t.Fatal(err)
	}
	registry := completionToolRegistry{sessionID: "session", handlers: map[string]registeredToolHandler{"failed": func(context.Context, llm.ToolCall, []llm.Message, eventEmitter) (registeredToolResult, error) {
		return registeredToolResult{Attachments: []db.Attachment{output}}, fmt.Errorf("worker failed")
	}}}
	if _, err := registry.execute(ctx, llm.ToolCall{Function: llm.FunctionCall{Name: "failed"}}, nil, nil); err == nil {
		t.Fatal("failure missing")
	}
	items, err = s.sessionImageAttachmentsForContext(ctx, "session")
	if err != nil || len(items) != 1 || items[original.ID].ID != original.ID {
		t.Fatalf("failed output became available: %v %v", items, err)
	}
}
