package main

import (
	"context"
	"errors"
	"github.com/chromedp/cdproto/cdp"
	"github.com/chromedp/chromedp"
	"strings"
	"testing"
	"time"
)

func TestRenderedContentReadsAuthorShadowRoots(t *testing.T) {
	text := func(value string) *cdp.Node { return &cdp.Node{NodeType: 3, NodeName: "#text", NodeValue: value} }
	root := &cdp.Node{NodeName: "HTML", Children: []*cdp.Node{
		{NodeName: "DIV", Children: []*cdp.Node{text("불러오는 중...")}},
		{NodeName: "ARTICLE", ShadowRoots: []*cdp.Node{{NodeName: "#document-fragment", ShadowRootType: cdp.ShadowRootTypeClosed, Children: []*cdp.Node{
			{NodeName: "STYLE", Children: []*cdp.Node{text("p { color: red; }")}},
			{NodeName: "P", Children: []*cdp.Node{text("첫 번째 문단")}},
			{NodeName: "P", Children: []*cdp.Node{text("두 번째 문단")}},
		}}}},
		{NodeName: "INPUT", ShadowRoots: []*cdp.Node{{ShadowRootType: cdp.ShadowRootTypeUserAgent, Children: []*cdp.Node{text("browser internals")}}}},
	}}
	got, pending := renderedContentSnapshot(root)
	if got != "첫 번째 문단\n두 번째 문단" || !pending {
		t.Fatalf("unexpected snapshot %q pending=%v", got, pending)
	}
	root.Children = root.Children[1:]
	got, pending = renderedContentSnapshot(root)
	if pending || strings.Contains(got, "color") || strings.Contains(got, "internals") {
		t.Fatalf("unwanted content %q pending=%v", got, pending)
	}
}

func TestBrowserPhaseBoundsBlockedActions(t *testing.T) {
	start := time.Now()
	err := browserPhase("HTML capture", 10*time.Millisecond, chromedp.ActionFunc(func(ctx context.Context) error { <-ctx.Done(); return ctx.Err() })).Do(context.Background())
	if !errors.Is(err, context.DeadlineExceeded) || !strings.Contains(err.Error(), "HTML capture") || time.Since(start) > time.Second {
		t.Fatalf("phase was not bounded: %v", err)
	}
}
func TestAutoModeDetectsLoadingInLongNavigation(t *testing.T) {
	if !hasLoadingIndicator(strings.Repeat("menu ", 100) + "\n불러오는 중...\ncomments") {
		t.Fatal("navigation length hid the loading indicator")
	}
	if hasLoadingIndicator("This article describes loading a file.") {
		t.Fatal("ordinary prose was classified as loading")
	}
}
