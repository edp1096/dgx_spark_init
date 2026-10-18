package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"sparktalk/internal/db"
	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

// Fixed multilingual controls: the source excerpts from Gift of the Magi and
// Kappa are from the previously frozen corpus. Other rows test adversarial
// homonyms, ordinary words and overlapping Latin names. No book-specific code.
func TestDocumentTermsFreezeControls(t *testing.T) {
	dir := os.Getenv("SPARKTALK_TERMS_FREEZE_CONTROLS")
	if dir == "" {
		t.Skip("explicit frozen controls")
	}
	pairs := [][2]string{
		{"Three times Della counted it.", "델라는 세 번이나 그것을 셌다."},
		{"So Della did it.", "델러는 그렇게 했다."},
		{"トックは河童仲間の詩人です。", "토크는 갓파들 사이의 시인이다."},
		{"トックは僕の顔を見ると、いつも微笑してこう言うのです。", "트럭은 내 얼굴을 보면 언제나 미소를 지으며 이렇게 말한다."},
		{"华盛顿是美国的首都。", "워싱턴은 미국의 수도이다."},
		{"代表团抵达华盛顿。", "대표단은 뉴욕에 도착했다."},
		{"May, the doctor, arrived at the clinic.", "의사 메이는 진료소에 도착했다."},
		{"The clinic opened in May.", "진료소는 5월에 문을 열었다."},
		{"The ship Aurora entered the harbour.", "배 오로라가 항구에 들어왔다."},
		{"Aurora, the singer, walked onto the stage.", "가수 아우로라가 무대에 올랐다."},
		{"Ann spoke with Anna near the annex.", "앤은 별관 근처에서 애나와 이야기했다."},
		{"Anna returned to the annex.", "애나는 별관으로 돌아왔다."},
	}
	source := []string{}
	for _, p := range pairs {
		source = append(source, p[0])
	}
	raw := []byte(strings.Join(source, "\n\n"))
	book, e := sourcebook.Parse(raw, "text/plain", "https://example.org/multilingual-controls", nil)
	if e != nil {
		t.Fatal(e)
	}
	database, e := db.Open(filepath.Join(dir, "before.db"))
	if e != nil {
		t.Fatal(e)
	}
	defer database.Close()
	sid, rid, sourceID := "terms-controls", "terms-controls-run", "terms-controls-source"
	if _, e = database.CreateSession(sid, "Multilingual consistency controls", "qwen38fn_exl3", "medium"); e != nil {
		t.Fatal(e)
	}
	if e = database.SaveSource(sourceID, sid, book, raw); e != nil {
		t.Fatal(e)
	}
	units := planDocumentTranslation(sourceID, book, 0, 0)
	values := map[string]string{}
	for i, b := range book.Blocks {
		values[b.ID] = pairs[i][1]
	}
	for i := range units {
		u := &units[i]
		u.Status = "completed"
		u.Output = map[string]string{}
		for _, in := range u.Inputs {
			u.Output[in.ID] = values[in.Block]
		}
	}
	glossary := map[string]string{}
	for _, name := range []string{"Della", "トック", "华盛顿", "May", "Aurora", "Ann"} {
		glossary[name] = "untrusted legacy proposal"
	}
	run := workflows.Run{ID: rid, SessionID: sid, Status: "completed", Current: 2, Document: &workflows.DocumentTask{Kind: "translate", Sources: []string{sourceID}, Phase: "completed", Total: len(units), Completed: len(units), Revision: 1, Glossary: glossary, RequestSpec: &workflows.DocumentRequestSpec{Origin: "api", UserText: "원문을 한국어로 충실하게 번역한다. 서로 다른 대상과 일반명사는 임의로 통일하지 않는다."}, Formats: []string{"txt", "md", "html", "zip"}}, Steps: []workflows.StepState{{Status: "completed"}, {Status: "completed"}}}
	if e = database.CreateDocumentRun(&run, units); e != nil {
		t.Fatal(e)
	}
	text, _ := book.Plain(values)
	if e = database.SaveDocumentOutput(sid, rid, "output.txt", "text/plain", text); e != nil {
		t.Fatal(e)
	}
	fixture := map[string]any{"session": sid, "run": rid, "forbidden": []string{"트럭은", "뉴욕에"}, "preserve": pairs[6:], "sources": []string{"frozen Gutenberg Gift of the Magi", "frozen Aozora Kappa", "synthetic Chinese and homonym controls"}}
	data, _ := json.MarshalIndent(fixture, "", "  ")
	if e = os.WriteFile(filepath.Join(dir, "fixture.json"), data, 0600); e != nil {
		t.Fatal(e)
	}
}
