package browserbridge

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestChromeExtension(t *testing.T) {
	if os.Getenv("TALK_BROWSER_E2E") != "1" {
		t.Skip("set TALK_BROWSER_E2E=1 to run Chromium fixture")
	}
	for _, mode := range []string{"inline", "trusted", "popup", "popup_window", "popup_reuse", "resume_source", "resume_popup", "resume_dom", "idle", "unmatched"} {
		t.Run(mode, func(t *testing.T) { runChromeExtension(t, mode) })
	}
}
func runChromeExtension(t *testing.T, mode string) {
	fixture := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if r.URL.Query().Get("resume_dom") == "1" {
			w.Write([]byte(`<div role="dialog" id="resume">작성 중인 리뷰가 있습니다. 이어서 작성하시겠습니까?<button id="resumeYes">이어서 작성</button><button>취소</button></div><script>document.querySelector('#resumeYes').onclick=e=>{if(!e.isTrusted)throw Error('Untrusted resume');document.querySelector('#resume').remove();};</script>`))
		}
		if r.URL.Query().Get("resume") == "1" {
			w.Write([]byte(`<script>window.resumed=confirm("작성 중이던 리뷰가 있습니다. 이어서 작성하시겠습니까?");if(!window.resumed)location.href="/cancelled";</script>`))
		}
		w.Write([]byte(`<!doctype html><meta charset="utf-8"><section><div><a role="button" aria-haspopup="true" aria-expanded="false" data-shp-area="rvw.pntop" href="#">리뷰 작성 시 포인트 <span>10원</span> 적립</a><h2>테스트 머그컵 리뷰 작성</h2></div><input type="radio" name="rating" aria-label="5점"><input type="radio" name="rating" aria-label="4점"><div class="wrapBox_wrap_box__test"><strong>품질은 어떤가요?</strong><div role="radiogroup"><a href="#" role="radio" aria-checked="false" data-value="1">보통이에요</a><a href="#" role="radio" aria-checked="false" data-value="2">좋아요</a></div></div><div class="Review_inner"><textarea></textarea></div><button id="submit">리뷰 등록</button></section><div role="status" id="notice"></div><script>document.querySelectorAll('[role="radio"]').forEach(el=>el.onclick=e=>{e.preventDefault();if(!e.isTrusted)throw Error("Untrusted evaluation");el.parentElement.querySelectorAll('[role="radio"]').forEach(x=>x.setAttribute("aria-checked",String(x===el)));});window.submissions=0;document.querySelector('input').onclick=e=>window.ratingTrusted=e.isTrusted;document.querySelector('textarea').oninput=e=>window.textTrusted=e.isTrusted;document.querySelector('#submit').onclick=e=>{if(!e.isTrusted||!window.ratingTrusted||!window.textTrusted)throw Error('Untrusted popup interaction');window.submissions++;document.querySelector('#notice').textContent='리뷰 등록이 완료되었습니다';};</script>`))
	}))
	defer fixture.Close()
	b, s := setup(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "node", "../../tests/browser/extension.cjs")
	cmd.Env = append(os.Environ(), "BRIDGE_URL="+s.URL, "BRIDGE_TOKEN="+b.token, "BROWSER_CASE="+mode, "FIXTURE_URL="+fixture.URL)
	cmd.Stderr = os.Stderr
	out, _ := cmd.StdoutPipe()
	in, _ := cmd.StdinPipe()
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer cmd.Process.Kill()
	scan := bufio.NewScanner(out)
	if !scan.Scan() || scan.Text() != "READY" {
		t.Fatal("extension failed to start")
	}
	for !b.Connected() {
		select {
		case <-ctx.Done():
			t.Fatal("connection timeout")
		case <-time.After(20 * time.Millisecond):
		}
	}
	call := func(action string, args any) json.RawMessage {
		t.Helper()
		v, e := b.Call(ctx, action, args)
		if e != nil {
			t.Fatal(e)
		}
		return v
	}
	var tabs struct {
		Tabs []struct {
			ID int `json:"tab_id"`
		} `json:"tabs"`
	}
	json.Unmarshal(call("tabs", nil), &tabs)
	expectedTabs := 1
	if mode == "idle" || mode == "unmatched" || mode == "popup_reuse" {
		expectedTabs = 2
	}
	if len(tabs.Tabs) != expectedTabs {
		t.Fatal(tabs)
	}
	var listing struct {
		Items []struct {
			ID      string `json:"id"`
			Product string `json:"product"`
		} `json:"items"`
	}
	raw := call("inspect", map[string]any{"tab_id": tabs.Tabs[0].ID})
	json.Unmarshal(raw, &listing)
	if len(listing.Items) != 2 {
		t.Fatalf("listing %s", raw)
	}
	drafts := []map[string]any{}
	for _, item := range listing.Items {
		drafts = append(drafts, map[string]any{"id": item.ID, "product": item.Product, "rating": 5, "text": "직접 사용한 소감입니다."})
	}
	args := map[string]any{"reviews": drafts}
	raw = call("open", map[string]any{"target_id": listing.Items[0].ID})
	if mode == "idle" || mode == "unmatched" {
		var failed struct {
			OK          bool `json:"ok"`
			Observation struct {
				Failure string `json:"failure"`
				Before  []any  `json:"before_tabs"`
				New     []any  `json:"new_tabs"`
				Changed []any  `json:"changed_tabs"`
			} `json:"observation"`
		}
		if json.Unmarshal(raw, &failed) != nil || failed.OK {
			t.Fatalf("missing failure %s", raw)
		}
		expected := "no_editor_or_navigation_observed"
		if mode == "unmatched" {
			expected = "editor_detected_but_unmatched"
		}
		if failed.Observation.Failure != expected || len(failed.Observation.Before) != 2 || len(failed.Observation.New) != 0 || len(failed.Observation.Changed) != 0 {
			t.Fatalf("incorrect attribution %s", raw)
		}
		in.Write([]byte("finish\n"))
		in.Close()
		for scan.Scan() {
			t.Log(scan.Text())
		}
		if err := cmd.Wait(); err != nil {
			t.Fatal(err)
		}
		return
	}
	if (mode == "resume_source" || (mode == "resume_popup" || mode == "resume_dom")) && !strings.Contains(string(raw), `"accepted":true`) {
		t.Fatalf("draft confirm not accepted: %s", raw)
	}
	if !strings.Contains(string(raw), `"status":"editor_open"`) {
		t.Fatalf("open failed %s", raw)
	}
	if !strings.Contains(string(raw), `"trusted_event":true`) || !strings.Contains(string(raw), `"user_activation":true`) {
		t.Fatalf("click was not trusted %s", raw)
	}
	popupTabID := 0
	if mode == "popup_window" || mode == "popup_reuse" || mode == "resume_source" || (mode == "resume_popup" || mode == "resume_dom") {
		var opened struct {
			TabID       int    `json:"tab_id"`
			WindowID    int    `json:"window_id"`
			WindowType  string `json:"window_type"`
			Observation struct {
				New []struct {
					Type string `json:"type"`
				} `json:"new_windows"`
				Reused []any `json:"reused_windows"`
			} `json:"observation"`
		}
		if json.Unmarshal(raw, &opened) != nil || opened.WindowType != "popup" || opened.WindowID == 0 {
			t.Fatalf("not a real popup window %s", raw)
		}
		popupTabID = opened.TabID
		if mode == "popup_window" && (len(opened.Observation.New) != 1 || opened.Observation.New[0].Type != "popup") {
			t.Fatalf("popup creation not recorded %s", raw)
		}
		if mode == "popup_reuse" && (len(opened.Observation.New) != 0 || len(opened.Observation.Reused) != 1) {
			t.Fatalf("popup reuse not recorded %s", raw)
		}
	}
	if mode == "popup_window" || mode == "popup_reuse" || mode == "popup" || mode == "resume_source" || (mode == "resume_popup" || mode == "resume_dom") {
		var form struct {
			Questions []struct {
				ID      string `json:"id"`
				Options []struct {
					ID string `json:"id"`
				} `json:"options"`
			} `json:"questions"`
		}
		if json.Unmarshal(raw, &form) != nil || len(form.Questions) != 1 || len(form.Questions[0].Options) != 2 {
			t.Fatalf("additional questions missing %s", raw)
		}
		drafts[0]["answers"] = []map[string]string{{"question_id": form.Questions[0].ID, "option_id": form.Questions[0].Options[1].ID}}
	}
	raw = call("fill", map[string]any{"reviews": drafts[:1]})
	if !strings.Contains(string(raw), `"status":"filled"`) || !strings.Contains(string(raw), `"submitted":false`) {
		t.Fatalf("fill failed %s", raw)
	}
	if mode == "popup_window" {
		blocked := call("close_popup", map[string]any{"tab_id": popupTabID})
		if !strings.Contains(string(blocked), "작성 중") {
			t.Fatalf("unsaved popup was not protected: %s", blocked)
		}
	}
	raw = call("read_current", map[string]any{"target_id": listing.Items[0].ID})
	var snapshot map[string]any
	if json.Unmarshal(raw, &snapshot) != nil || snapshot["status"] != "current_form" {
		t.Fatalf("current read %s", raw)
	}
	if mode == "popup_window" {
		in.Write([]byte("edit\n"))
		if !scan.Scan() || scan.Text() != "EDITED" {
			t.Fatal("manual edit fixture failed")
		}
		stale := call("submit_current", map[string]any{"target_id": listing.Items[0].ID, "snapshot": snapshot})
		if !strings.Contains(string(stale), `"attempted_submit":false`) {
			t.Fatalf("stale snapshot was not blocked %s", stale)
		}
		raw = call("read_current", map[string]any{"target_id": listing.Items[0].ID})
		json.Unmarshal(raw, &snapshot)
		if snapshot["text"] != "사용자가 팝업에서 고친 최종 후기" || snapshot["rating"] != float64(4) {
			t.Fatalf("manual edits overwritten %s", raw)
		}
	}
	raw = call("submit_current", map[string]any{"target_id": listing.Items[0].ID, "snapshot": snapshot})
	if !strings.Contains(string(raw), `"status":"submitted"`) {
		t.Fatalf("current submission %s", raw)
	}
	raw = call("submit", map[string]any{"reviews": drafts[1:]})
	if !strings.Contains(string(raw), `"status":"submitted"`) {
		t.Fatalf("submission %s", raw)
	}
	raw = call("submit", args)
	if strings.Contains(string(raw), `"status":"submitted"`) || !strings.Contains(string(raw), "중복") {
		t.Fatalf("duplicate not blocked %s", raw)
	}
	if mode == "popup_window" {
		in.Write([]byte("close_ready\n"))
		if !scan.Scan() || scan.Text() != "CLOSE_READY" {
			t.Fatal("close fixture failed")
		}
		closed := call("close_popup", map[string]any{"tab_id": popupTabID})
		if !strings.Contains(string(closed), `"status":"popup_closed"`) {
			t.Fatalf("close failed %s", closed)
		}
		again := call("close_popup", map[string]any{"tab_id": popupTabID})
		if !strings.Contains(string(again), `"status":"already_closed"`) {
			t.Fatalf("close retry failed %s", again)
		}
		mainID := tabs.Tabs[0].ID
		blocked := call("close_popup", map[string]any{"tab_id": mainID})
		if !strings.Contains(string(blocked), "일반 탭") {
			t.Fatalf("normal tab not protected %s", blocked)
		}
		refreshed := call("refresh", map[string]any{"tab_id": mainID})
		if !strings.Contains(string(refreshed), `"status":"refreshed"`) {
			t.Fatalf("refresh failed %s", refreshed)
		}
		fresh := call("inspect", map[string]any{"tab_id": mainID})
		if !strings.Contains(string(fresh), "테스트 머그컵") {
			t.Fatalf("reload inspection failed %s", fresh)
		}
	}
	in.Write([]byte("finish\n"))
	in.Close()
	for scan.Scan() {
		t.Log(scan.Text())
	}
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}
}
