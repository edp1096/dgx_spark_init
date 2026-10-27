package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/orchestrator"
)

// Optional read-only model check: tool calls are inspected, never executed.
func TestShortsNaturalLanguageRoutingLive(t *testing.T) {
	endpoint := os.Getenv("TALK_SHORTS_MODEL_ENDPOINT")
	if endpoint == "" {
		t.Skip("set TALK_SHORTS_MODEL_ENDPOINT for the local model routing check")
	}
	s, reg := shortsToolTestServer(t, http.NotFound)
	s.registerVideoTool(context.Background(), &reg, s.cfg, func(db.Attachment) error { return nil })
	client := llm.New(endpoint, "qwen38fn_exl3", "", "qwen3.8-tensorfold")
	for _, tc := range []struct{ name, context, prompt, tool, action string }{
		{"voices", "", "fish에는 영상생성에 사용가능한 TTS 음성이 몇명 그리고 뭐뭐 있냐?", "shorts_voices", "list"},
		{"voice_preview", "Installed voice_id: ryan.", "Ryan 목소리만 들어볼래. 영상 만들지 마.", "shorts_voices", "preview"},
		{"voice_select", "Installed voice_id: ryan.", "앞으로 만드는 영상의 기본 음성을 Ryan으로 선택해.", "shorts_voices", "select"},
		{"voice_apply", "The latest video is completed. Installed voice_id: ryan.", "기존 영상에 Ryan 음성을 적용해. 화면은 그대로 둬.", "shorts_voices", "apply"},
		{"create", "", "캐시 원리를 설명하는 30초짜리 한국어 쇼츠를 만들어줘. 자막과 잔잔한 브금도 넣어.", "shorts_generate", ""},
		{"music", "", "ACE-Step XL로 잔잔한 피아노 연주곡 30초 만들어줘. 보컬 없이 음악 파일만 줘.", "music_generate", ""},
		{"status", "A shorts job is running in this conversation.", "지금 쇼츠 얼마나 진행됐어?", "shorts_manage", "status"},
		{"cancel", "A shorts job is running in this conversation.", "그 쇼츠 제작 중지해.", "shorts_manage", "cancel"},
		{"retry", "The latest shorts job failed while rendering.", "아까 실패한 쇼츠 다시 시도해.", "shorts_manage", "retry"},
		{"assets", "", "LottieFiles에서 무료 하트 애니메이션을 검색해줘. 아직 영상은 만들지 마.", "shorts_assets", "search"},
		{"templates", "", "이 대화에 저장된 쇼츠 템플릿 목록 보여줘.", "shorts_templates", "list"},
		{"revise", "The latest short is completed and has reusable photo assets.", "방금 쇼츠의 사진은 재사용하고 자막을 중앙 흰색으로 고쳐서 미리보기 만들어줘.", "shorts_project", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			messages := []llm.Message{{Role: "system", Content: strings.Join(reg.prompts, "\n") + "\n" + tc.context}, {Role: "user", Content: tc.prompt}}
			result, err := client.Stream(llm.WithOutputLimit(ctx, 700), messages, "qwen38fn_exl3", "none", reg.definitions, func(string, string) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
			if len(result.ToolCalls) != 1 || result.ToolCalls[0].Function.Name != tc.tool {
				t.Fatalf("wrong routing: %+v / %s", result.ToolCalls, result.Content)
			}
			if tc.action != "" {
				var args struct {
					Action string `json:"action"`
				}
				_ = json.Unmarshal([]byte(result.ToolCalls[0].Function.Arguments), &args)
				if args.Action != tc.action {
					t.Fatal("wrong action", args.Action)
				}
			}
			t.Log(tc.prompt + " → " + result.ToolCalls[0].Function.Name + " " + result.ToolCalls[0].Function.Arguments)
		})
	}
}

func shortsToolTestServer(t *testing.T, handler http.HandlerFunc) (*Server, completionToolRegistry) {
	t.Helper()
	worker := httptest.NewServer(handler)
	t.Cleanup(worker.Close)
	s, _ := testImageServer(t)
	cat, _ := orchestrator.LoadCatalog()
	for i := range cat.Components {
		if cat.Components[i].ID == "extra-shorts" {
			cat.Components[i].Endpoint, cat.Components[i].HealthURL = worker.URL, worker.URL+"/health"
			cat.Components[i].Controller, cat.Components[i].AutoAddress = "external", false
		}
	}
	runtime, err := orchestrator.NewControllerWithCatalog(cat)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(runtime.Close)
	s.runtime = runtime
	s.cfg.Runtime.Mode, s.cfg.Runtime.ActiveBundle, s.cfg.Runtime.Catalog = "managed", "qwen38fn_exl3", &cat
	reg := completionToolRegistry{sessionID: "session", handlers: map[string]registeredToolHandler{}}
	s.registerShortsTool(&reg)
	return s, reg
}

func TestShortsChatGenerateStreamsProgressAndPublishesOnce(t *testing.T) {
	jobID := "abcdef0123456789abcdef0123456789"
	var reads, creates atomic.Int32
	s, reg := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			io.WriteString(w, `{"status":"ok"}`)
			return
		}
		if strings.HasSuffix(r.URL.Path, "/final.mp4") {
			w.Header().Set("Content-Type", "video/mp4")
			io.WriteString(w, "\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isommp42")
			return
		}
		job := shortsJob{ID: jobID, SessionID: "session", Title: "캐시 설명", Status: "queued", Stage: "대기 중", Created: float64(time.Now().Unix())}
		if r.Method == "POST" {
			creates.Add(1)
			var a shortsRequest
			_ = json.NewDecoder(r.Body).Decode(&a)
			if a.SessionID != "session" || !a.BGM || a.Brief != "캐시 원리를 한국어 쇼츠로 만들어줘" {
				t.Error("conversation request lost", a)
			}
		} else if reads.Add(1) == 1 {
			job.Status, job.Stage, job.Progress = "running", "Fish 음성 준비·합성", 15
		} else {
			job.Status, job.Stage, job.Progress, job.OutputReady = "completed", "완료", 100, true
		}
		_ = json.NewEncoder(w).Encode(job)
	})
	reply, err := s.db.AddMessage("session", "assistant", "이전 실패", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	before, _ := s.db.Messages("session")
	var progress []generationProgress
	result, err := reg.handlers["shorts_generate"](context.Background(), llm.ToolCall{ID: "shorts", Function: llm.FunctionCall{Arguments: `{"title":"캐시 설명","brief":"캐시 원리를 한국어 쇼츠로 만들어줘","bgm":true}`}}, nil, func(event string, payload any) error {
		if event == "tool_output" {
			b, _ := json.Marshal(payload)
			var v struct {
				Progress generationProgress `json:"progress"`
			}
			_ = json.Unmarshal(b, &v)
			progress = append(progress, v.Progress)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if creates.Load() != 1 || len(progress) < 3 || progress[1].Detail != "Fish 음성 준비·합성" || !strings.Contains(result.Result, `"delivery"`) {
		t.Fatal("missing chat progress/delivery", creates.Load(), progress, result.Result)
	}
	after, _ := s.db.Messages("session")
	if len(after) != len(before)+1 || len(after[len(after)-1].Attachments) != 1 {
		t.Fatal("video not attached exactly once")
	}
	if _, err = s.shortsPublish(context.Background(), shortsRequest{SessionID: "session", JobID: jobID}); err != nil {
		t.Fatal(err)
	}
	afterAgain, _ := s.db.Messages("session")
	if len(afterAgain) != len(after) {
		t.Fatal("worker callback duplicated tool publication")
	}
	trace := []db.ToolEvent{{Name: "shorts_generate", Result: result.Result}}
	if err := s.db.ReplaceAssistantWithAttachments(reply.ID, "완료", "", trace, nil, 0); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		if err := s.restoreShortsDeliveries(context.Background(), "session", trace); err != nil {
			t.Fatal(err)
		}
	}
	restored, _ := s.db.Messages("session")
	if len(restored) != len(after) || len(restored[len(restored)-1].Attachments) != 1 {
		t.Fatal("retry lost or duplicated the asynchronous MP4")
	}
}

func TestShortsChatManageCannotControlAnotherConversation(t *testing.T) {
	jobID := "abcdef0123456789abcdef0123456789"
	var actions atomic.Int32
	_, reg := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			io.WriteString(w, `{"status":"ok"}`)
			return
		}
		if r.Method == "POST" {
			actions.Add(1)
		}
		_ = json.NewEncoder(w).Encode(shortsJob{ID: jobID, SessionID: "other", Status: "running"})
	})
	for _, action := range []string{"status", "wait", "cancel", "retry"} {
		args, _ := json.Marshal(map[string]string{"action": action, "job_id": jobID})
		if _, err := reg.handlers["shorts_manage"](context.Background(), llm.ToolCall{Function: llm.FunctionCall{Arguments: string(args)}}, nil, nil); err == nil {
			t.Fatal("foreign job accepted", action)
		}
	}
	if actions.Load() != 0 {
		t.Fatal("foreign job mutated")
	}
}

func TestShortsWaitYieldsToChatInputWithoutCancelingDurableJob(t *testing.T) {
	s, _ := testImageServer(t)
	ctx := context.WithValue(context.Background(), activeTurnKey{}, &activeTurn{pending: []db.TurnInput{{Content: "그 쇼츠 중지해"}}})
	job := shortsJob{ID: "abcdef0123456789abcdef0123456789", SessionID: "session", Status: "running", Stage: "합성 중"}
	result, err := s.waitForShorts(ctx, "session", "shorts", job, nil)
	if err != nil || !strings.Contains(result.Result, job.ID) || !strings.Contains(result.Result, `"running"`) {
		t.Fatal("new input cannot reach model with active job", result, err)
	}
}

func TestOnlyExplicitChatStopCancelsDurableShorts(t *testing.T) {
	for _, explicit := range []bool{false, true} {
		var stops atomic.Int32
		s, _ := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" && strings.HasSuffix(r.URL.Path, "/cancel") {
				stops.Add(1)
			}
			io.WriteString(w, `{}`)
		})
		ctx, cancel := context.WithCancelCause(context.Background())
		if explicit {
			cancel(errChatStopped)
		} else {
			cancel(context.Canceled)
		}
		job := shortsJob{ID: "abcdef0123456789abcdef0123456789", SessionID: "session", Status: "running"}
		if _, err := s.waitForShorts(ctx, "session", "shorts", job, nil); err == nil {
			t.Fatal("canceled wait succeeded")
		}
		want := int32(0)
		if explicit {
			want = 1
		}
		if stops.Load() != want {
			t.Fatal("wrong durable cancellation", explicit, stops.Load())
		}
	}
}

func TestShortsChatCancelAndRetryLatestJob(t *testing.T) {
	job := shortsJob{ID: "abcdef0123456789abcdef0123456789", SessionID: "session", Status: "running", Stage: "합성 중"}
	var mutations []string
	_, reg := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			io.WriteString(w, `{"status":"ok"}`)
			return
		}
		if r.Method == "POST" {
			mutations = append(mutations, r.URL.Path)
			if strings.HasSuffix(r.URL.Path, "/cancel") {
				job.Status = "canceled"
			} else if strings.HasSuffix(r.URL.Path, "/retry") {
				job.Status = "queued"
			} else {
				t.Error("unexpected create", r.URL.Path)
			}
		}
		if r.URL.Path == "/v1/projects" {
			if r.URL.Query().Get("session_id") != "session" {
				t.Error("missing conversation scope")
			}
			_ = json.NewEncoder(w).Encode([]shortsJob{job})
			return
		}
		_ = json.NewEncoder(w).Encode(job)
	})
	ctx := context.WithValue(context.Background(), activeTurnKey{}, &activeTurn{pending: []db.TurnInput{{Content: "진행 상황 알려줘"}}})
	for _, action := range []string{"cancel", "retry"} {
		if _, err := reg.handlers["shorts_manage"](ctx, llm.ToolCall{Function: llm.FunctionCall{Arguments: `{"action":"` + action + `"}`}}, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	if len(mutations) != 2 || !strings.HasSuffix(mutations[0], job.ID+"/cancel") || !strings.HasSuffix(mutations[1], job.ID+"/retry") {
		t.Fatal("latest job not reused", mutations)
	}
}
