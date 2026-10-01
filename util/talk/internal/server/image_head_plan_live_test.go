package server

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
	"sparktalk/internal/orchestrator"
)

func TestLiveHeadPlanEndToEnd(t *testing.T) {
	if os.Getenv("TALK_LIVE_HEAD_PLAN") != "1" {
		t.Skip("explicit actual Qwen and BFS workflow")
	}
	cfg, _, err := config.Load("../../dist/sparktalk.yaml")
	if err != nil {
		t.Fatal(err)
	}
	s, _ := testImageServer(t)
	if os.Getenv("TALK_LIVE_HEAD_PLAN_PUBLISH") == "1" {
		store, e := db.Open("../../dist/sparktalk.db")
		if e != nil {
			t.Fatal(e)
		}
		defer store.Close()
		ms, e := media.New("../../dist/sparktalk.db")
		if e != nil {
			t.Fatal(e)
		}
		s = &Server{db: store, media: ms}
	}
	staged := os.Getenv("TALK_LIVE_IMAGE_STAGES") == "1"
	grouped := os.Getenv("TALK_LIVE_REFERENCE_GROUPS") == "1"
	if staged {
		cfg.Tools.SkillsEnabled = true
		cfg.Tools.Enabled = true
	}
	s.cfg = cfg
	s.runtime, err = orchestrator.NewControllerWithCatalog(*cfg.Runtime.Catalog)
	if err != nil {
		t.Fatal(err)
	}
	s.runtime.ConfigurePaths(cfg.Runtime.DataDir, cfg.Runtime.ModelCache)
	session := fmt.Sprintf("bfs-plan-check-%d", time.Now().UnixNano())
	if _, err = s.db.CreateSession(session, func() string {
		if grouped {
			return "인물별 사진 6장 검증 · " + time.Now().Format("15:04")
		}
		if staged {
			return "이미지 단계별 제작 검증 · " + time.Now().Format("15:04")
		}
		return "BFS 자동 보정 검증 · " + time.Now().Format("15:04")
	}(), cfg.Model.DefaultModel, "none"); err != nil {
		t.Fatal(err)
	}
	photos := []db.Attachment{}
	ids := []string{"e7d1228f6aea7770287712745a4d2505", "a4fb1acedcb5eab3f01af2d2f858355f"}
	if grouped {
		ids = []string{"073ede2be476fb36917169c9be682809", "b52217193707e2316abab647857c4770", "8c1dae78b9e5159e638c5d8db937786f", "6cdd805d4171c0d1b6f8690d5169d657", "2fa9c7130593c1c9f0c251f4adfc80c0", "e620d15be0c4c9db8083575c3cff1205"}
	}
	for _, id := range ids {
		f, e := os.Open(filepath.Join("../../dist/sparktalk.db.media", id))
		if e != nil {
			t.Fatal(e)
		}
		a, e := s.media.SaveReader(f, id+".jpg", "image/jpeg", media.MaxImageBytes)
		f.Close()
		if e != nil {
			t.Fatal(e)
		}
		photos = append(photos, a)
	}
	prompt := "[자동 검증 요청] 이미지1: 덩샤오핑, 이미지2: 시진핑이다. 두 사진의 얼굴 특징을 반영해 덩샤오핑이 시진핑을 신문으로 때리는 장면을 만화풍으로 그려라. 인물은 정확히 두 명이고 때리는 동작이 보여야 한다. 사진 배경은 무시해라."
	if staged {
		prompt = "[자동 검증 요청] 이미지1은 덩샤오핑, 이미지2는 시진핑이다. 두 인물이 나란히 서서 손을 흔드는 전신 만화로 그려라. 덩샤오핑은 왼쪽 인민복, 시진핑은 오른쪽 남색 양복과 파란 넥타이. 원본 얼굴 특징과 인물 구분을 유지하고 배경은 연한 하늘색으로 해라."
	}
	if grouped {
		prompt = "[자동 검증 요청] 이미지1,2,3은 등소평, 이미지4,5,6은 시진핑이다. 인물별 사진을 함께 참고해 등소평이 신문으로 시진핑의 머리를 때리는 장면을 만화풍으로 그려라. 인물은 두 명이며 사진 배경은 무시해라. 사진 6은 현재 요청에 저장된 첨부가 5장이라 같은 대화의 기존 시진핑 원본을 보완한 것이다."
	}
	if _, err = s.db.AddMessage(session, "user", prompt, "", nil, photos); err != nil {
		t.Fatal(err)
	}
	messages, err := s.llmMessages(context.Background(), []db.Message{{Role: "user", Content: prompt, Attachments: photos}}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Minute)
	defer cancel()
	markerMessages := messages
	toolsConfig := config.ToolsConfig{Enabled: true, MaxRounds: 8, Timeout: "120s", SkillsEnabled: staged}
	if staged {
		markerMessages = append(append([]llm.Message{}, messages...), llm.Message{Role: "user", Content: "@workflow:identity-image-production\n" + prompt})
	}
	result, runErr := runCompletionLoopForSessionWithMedia(s, session, ctx, llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType), markerMessages, cfg.Model.DefaultModel, "none", "", toolsConfig, true, func(kind string, p any) error {
		if kind == "tool_start" || kind == "tool_result" || kind == "workflow" {
			b, _ := json.Marshal(p)
			t.Logf("%s %s", kind, b)
		}
		return nil
	}, func(db.Attachment) error { return nil })
	note := "자동 검증 실행 기록입니다. 중간 생성과 BFS 보정 결과가 순서대로 첨부됩니다. 얼굴 닮음과 동작의 완성도는 이미지로 확인해야 합니다.\n\n" + result.Content
	if runErr != nil {
		note += "\n검증 오류: " + runErr.Error()
	}
	if _, err = s.db.AddMessage(session, "assistant", note, result.Reasoning, result.ToolTrace, result.Attachments); err != nil {
		t.Fatal(err)
	}
	t.Log("Talk session:", session)
	if grouped {
		found := false
		for _, e := range result.ToolTrace {
			if e.Name != "image_generate" || e.Error != "" {
				continue
			}
			var r struct {
				Groups []imageReferenceGroupEvidence `json:"reference_groups"`
				Inputs []imageInputEvidence          `json:"input_images"`
			}
			json.Unmarshal([]byte(e.Result), &r)
			if len(r.Groups) == 2 && len(r.Inputs) == 2 && len(r.Groups[0].Candidates) == 3 && len(r.Groups[1].Candidates) == 3 {
				found = true
			}
		}
		if !found {
			t.Fatal("six photos were not preserved as two subject groups")
		}
	}
	n := 0
	for _, e := range result.ToolTrace {
		var a imageGenerationArgs
		json.Unmarshal([]byte(e.Arguments), &a)
		if e.Name == "image_generate" && a.Operation == "head_swap" && e.Error == "" {
			n++
		}
	}
	if runErr != nil {
		t.Fatal(runErr)
	}
	if n < 2 {
		t.Fatalf("head correction plan skipped: only %d successful head calls", n)
	}
	t.Logf("actual head corrections=%d outputs=%d", n, len(result.Attachments))
}
