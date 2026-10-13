package server

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

// Recover authority from saved user messages, never from the assistant's old
// tool arguments. An untraceable legacy request requires an explicit revision.
func (s *Server) documentRevisionRequest(parent workflows.Run) (*workflows.DocumentRequestSpec, error) {
	if spec := parent.Document.RequestSpec; spec != nil {
		copy := *spec
		copy.MessageIDs = append([]int64(nil), spec.MessageIDs...)
		return &copy, nil
	}
	messages, err := s.db.Messages(parent.SessionID)
	if err != nil {
		return nil, err
	}
	anchor, _ := strconv.ParseInt(strings.SplitN(parent.Document.RequestID, "/", 2)[0], 10, 64)
	var lastUserID int64
	var lastUserText string
	for _, message := range messages {
		if message.Role == "user" {
			lastUserID, lastUserText = message.ID, userTurnContent(message)
			if message.ID == anchor {
				return &workflows.DocumentRequestSpec{Origin: "recovered_user_turn", MessageIDs: []int64{message.ID}, UserText: lastUserText, Hash: documentTextHash(lastUserText), ToolHints: parent.Document.Instructions}, nil
			}
		}
		// The immutable run ID in a tool result identifies the originating turn.
		for _, event := range message.ToolTrace {
			var result struct {
				RunID string `json:"run_id"`
			}
			if event.Name == "document_task" && json.Unmarshal([]byte(event.Result), &result) == nil && result.RunID == parent.ID && lastUserID > 0 {
				return &workflows.DocumentRequestSpec{Origin: "recovered_user_turn", MessageIDs: []int64{lastUserID}, UserText: lastUserText, Hash: documentTextHash(lastUserText), ToolHints: parent.Document.Instructions}, nil
			}
		}
	}
	return nil, fmt.Errorf("이전 작업의 실제 사용자 지시를 복원할 수 없습니다. 새 지시로 개정판을 만들어 주세요")
}

func documentMapRevisionOffset(oldBook, fresh *sourcebook.Book, offset int) (int, error) {
	if offset == 0 {
		return 0, nil
	}
	oldPaths, newPaths := documentBlockPaths(oldBook), documentBlockPaths(fresh)
	byPath := map[string]sourcebook.Block{}
	for _, b := range fresh.Blocks {
		byPath[newPaths[b.ID]] = b
	}
	for _, b := range oldBook.Blocks {
		if offset >= b.Start && offset <= b.End+1 {
			if n, ok := byPath[oldPaths[b.ID]]; ok && b.Text == n.Text {
				return n.Start + offset - b.Start, nil
			}
		}
	}
	return 0, fmt.Errorf("선택 범위의 원문 좌표가 새 파서와 일치하지 않습니다. 새 개정판의 범위를 지정해 주세요")
}

// Called with docTaskMu held. Creation has no effect on the parent's checkpoints,
// source snapshot or downloads; launching happens after releasing that lock.
func (s *Server) createDocumentQualityRevision(parent workflows.Run) (workflows.Run, error) {
	var run workflows.Run
	if len(parent.Document.Sources) != 1 {
		return run, fmt.Errorf("quality revision requires one translation source")
	}
	spec, err := s.documentRevisionRequest(parent)
	if err != nil {
		return run, err
	}
	oldID := parent.Document.Sources[0]
	oldBook, err := s.db.Source(parent.SessionID, oldID, true)
	if err != nil {
		return run, err
	}
	prior, err := s.db.DocumentUnits(parent.SessionID, parent.ID)
	if err != nil {
		return run, err
	}
	fresh, newID := oldBook, oldID
	start, end := parent.Document.Start, parent.Document.End
	if oldBook.Version < sourcebook.SnapshotVersion {
		raw, e := s.db.RawSource(parent.SessionID, oldID)
		if e != nil {
			return run, e
		}
		if len(raw) > 0 && !strings.HasPrefix(string(raw), "%PDF-") {
			mime := http.DetectContentType(raw)
			loader := func(url string) ([]byte, string, error) {
				for _, asset := range oldBook.Assets {
					if asset.URL == url && len(asset.Data) > 0 && asset.Error == "" {
						return asset.Data, asset.MIME, nil
					}
				}
				return nil, "", fmt.Errorf("cached illustration unavailable")
			}
			fresh, e = sourcebook.Parse(raw, mime, oldBook.URL, loader)
			if e != nil {
				return run, e
			}
			if start, e = documentMapRevisionOffset(oldBook, fresh, start); e != nil {
				return run, e
			}
			if end, e = documentMapRevisionOffset(oldBook, fresh, end); e != nil {
				return run, e
			}
			if fresh.Metadata == nil {
				fresh.Metadata = map[string]any{}
			}
			fresh.Metadata["parent_source_id"] = oldID
			fresh.Metadata["parent_source_hash"] = oldBook.Hash
			newID, e = s.storeSource(parent.SessionID, fresh, raw)
			if e != nil {
				return run, e
			}
		}
	}
	planned := planDocumentTranslation(newID, fresh, start, end)
	units := carryDocumentTranslation(oldID, newID, oldBook, fresh, prior, planned)
	if len(units) == 0 {
		return run, fmt.Errorf("revision has no source text")
	}
	cfg, _ := s.snapshot()
	d := &workflows.DocumentTask{Pipeline: workflows.DocumentPipelineVersion, RequestSpec: spec, Kind: "translate", Sources: []string{newID}, SourceURLs: map[string]string{newID: fresh.URL}, SourceTitles: map[string]string{newID: fresh.Title}, Model: cfg.Model.DefaultModel, ModelType: cfg.Model.ModelType, Endpoint: cfg.Model.Endpoint, Effort: parent.Document.Effort, Instructions: spec.UserText, Revision: parent.Document.Revision + 1, Parent: parent.ID, Start: start, End: end, UserGlossary: parent.Document.UserGlossary, RequestedFormat: parent.Document.RequestedFormat, Formats: []string{"txt", "md", "html", "zip"}, Phase: "review", MaxCalls: 2048, MaxOutputTokens: 2_000_000, MaxSeconds: 12 * 60 * 60, OutputLimit: cfg.Context.EffectiveOutputReserve(cfg.Context.WindowTokens)}
	// An old inferred glossary had no authoritative provenance.
	if parent.Document.RequestSpec == nil {
		d.UserGlossary = nil
	}
	d.Profiles = documentProfiles(cfg, d.Effort)
	run = workflows.Run{ID: "document-" + rand.Text(), SessionID: parent.SessionID, Status: "paused", Goal: d.Instructions, Definition: parent.Definition, Document: d, Steps: []workflows.StepState{{Status: "pending"}, {Status: "pending"}}}
	if parent.Document.Pipeline == "" {
		// The existing missing-fragment action must not silently opt a legacy
		// document into an unqualified semantic reviewer. Keep its processing
		// contract while retaining the original output as a separate revision.
		d.Pipeline = ""
		d.Profiles = nil
		units, err = prepareDocumentQualityRepair(&run, units)
		if err != nil {
			return workflows.Run{}, err
		}
	} else {
		documentQualityCounters(&run, units)
	}
	return run, s.db.CreateDocumentRun(&run, units)
}

// DOM locations let a new parser revision retain unchanged translations even
// if an encoded character changes subsequent block IDs/absolute offsets.
func documentBlockPaths(book *sourcebook.Book) map[string]string {
	paths := map[string]string{}
	var visit func(*sourcebook.Node, string)
	visit = func(node *sourcebook.Node, path string) {
		if node == nil {
			return
		}
		path += "/" + node.Tag
		if node.Block != "" {
			paths[node.Block] = path + "/text"
		}
		if node.Alt != "" {
			paths[node.Alt] = path + "/alt"
		}
		for i, child := range node.Children {
			visit(child, fmt.Sprintf("%s/%d", path, i))
		}
	}
	visit(book.Root, "")
	return paths
}

// Reuse only translations covering exactly the same source spans. Never cut a
// target sentence to guess an alignment with a newly split source paragraph.
func carryDocumentTranslation(oldSource, newSource string, oldBook, newBook *sourcebook.Book, prior, planned []workflows.DocumentUnit) []workflows.DocumentUnit {
	oldPaths, newPaths := documentBlockPaths(oldBook), documentBlockPaths(newBook)
	oldByPath := map[string]sourcebook.Block{}
	for _, block := range oldBook.Blocks {
		if path := oldPaths[block.ID]; path != "" {
			oldByPath[path] = block
		}
	}
	type fragment struct {
		input  workflows.DocumentInput
		output string
	}
	fragments := map[string][]fragment{}
	for _, unit := range prior {
		for _, in := range unit.Inputs {
			if in.Source != oldSource {
				continue
			}
			if output, ok := unit.Output[in.ID]; ok {
				fragments[in.Block] = append(fragments[in.Block], fragment{in, output})
			}
		}
	}
	for block := range fragments {
		sort.Slice(fragments[block], func(i, j int) bool { return fragments[block][i].input.Part < fragments[block][j].input.Part })
	}
	newBlocks := map[string]sourcebook.Block{}
	for _, block := range newBook.Blocks {
		newBlocks[block.ID] = block
	}
	var out []workflows.DocumentUnit
	for _, unit := range planned {
		copied := map[string]string{}
		for _, in := range unit.Inputs {
			if in.Source != newSource {
				continue
			}
			fresh := newBlocks[in.Block]
			old, ok := oldByPath[newPaths[in.Block]]
			if !ok || old.Text != fresh.Text || old.Kind != fresh.Kind {
				continue
			}
			at, end := in.Part, in.Part+utf8.RuneCountInString(in.Text)
			var text strings.Builder
			for _, part := range fragments[old.ID] {
				partEnd := part.input.Part + utf8.RuneCountInString(part.input.Text)
				if partEnd <= at {
					continue
				}
				if part.input.Part != at || partEnd > end {
					break
				}
				source := []rune(fresh.Text)
				if at < 0 || partEnd > len(source) || string(source[at:partEnd]) != part.input.Text {
					break
				}
				text.WriteString(part.output)
				at = partEnd
				if at == end {
					copied[in.ID] = text.String()
					break
				}
			}
		}
		// A changed glyph or unaligned span invalidates its whole structural
		// paragraph. Completed neighbouring paragraphs retain their exact text.
		var current *workflows.DocumentUnit
		for _, group := range documentTextGroups(unit) {
			complete := true
			for _, in := range group.Inputs {
				_, ok := copied[in.ID]
				complete = complete && ok
			}
			status := "pending"
			if complete {
				status = "completed"
			}
			if current == nil || current.Status != status {
				out = append(out, workflows.DocumentUnit{Ordinal: len(out), Stage: "process", Status: status, Output: map[string]string{}, Quality: &workflows.DocumentUnitQuality{Status: "pending"}})
				current = &out[len(out)-1]
			}
			current.Inputs = append(current.Inputs, group.Inputs...)
			if complete {
				for _, in := range group.Inputs {
					current.Output[in.ID] = copied[in.ID]
				}
			}
		}
	}
	return out
}

func (s *Server) createDocumentAssetRevision(parent workflows.Run) (workflows.Run, error) {
	var child workflows.Run
	if parent.Document.Kind != "translate" || len(parent.Document.Sources) != 1 {
		return child, fmt.Errorf("illustration revision requires one translation source")
	}
	oldID := parent.Document.Sources[0]
	book, err := s.db.Source(parent.SessionID, oldID, true)
	if err != nil {
		return child, err
	}
	retry := 0
	for i := range book.Assets {
		a := &book.Assets[i]
		if a.Pending || a.Error != "" {
			a.Pending = true
			a.Error = ""
			a.Data = nil
			retry++
		}
	}
	if retry == 0 {
		return child, fmt.Errorf("no missing illustrations to retry")
	}
	raw, err := s.db.RawSource(parent.SessionID, oldID)
	if err != nil {
		return child, err
	}
	newSource, err := s.storeSource(parent.SessionID, book, raw)
	if err != nil {
		return child, err
	}
	units, err := s.db.DocumentUnits(parent.SessionID, parent.ID)
	if err != nil {
		return child, err
	}
	clone, _ := json.Marshal(parent)
	if err = json.Unmarshal(clone, &child); err != nil {
		return child, err
	}
	child.ID = "document-" + rand.Text()
	child.Status = "paused"
	child.Current = 0
	child.Updated = ""
	d := child.Document
	d.Parent = parent.ID
	d.Revision++
	d.OutputOnly = true
	d.Sources = []string{newSource}
	d.SourceURLs = map[string]string{newSource: book.URL}
	d.SourceTitles = map[string]string{newSource: book.Title}
	d.Publication = nil
	d.DraftPublication = nil
	d.Terminal = false
	d.FinishedAt = ""
	d.ElapsedSeconds = 0
	d.Calls = 0
	d.OutputTokens = 0
	d.StageMetrics = nil
	d.ExecutionID = ""
	d.Phase = "export"
	d.Error = ""
	d.Formats = []string{"txt", "md", "html", "zip"}
	d.Warnings = nil
	d.Performance = nil
	d.LastPerformance = nil
	d.RequestID = ""
	d.RequestHash = ""
	d.ActiveUnits = nil
	d.ImageTotal = len(book.Assets)
	d.ImagesReady = 0
	for _, asset := range book.Assets {
		if !asset.Pending && asset.Error == "" {
			d.ImagesReady++
		}
	}
	for i := range units {
		u := &units[i]
		if u.Status != "completed" {
			return child, fmt.Errorf("incomplete translation cannot be used for illustration-only revision")
		}
		ids := map[string]string{}
		output := map[string]string{}
		for j := range u.Inputs {
			in := &u.Inputs[j]
			old := in.ID
			in.Source = newSource
			in.ID = fmt.Sprintf("%s:%s:%d", newSource, in.Block, in.Part)
			ids[old] = in.ID
			output[in.ID] = u.Output[old]
		}
		u.Output = output
		u.Attempts = nil
		if u.Quality != nil {
			u.Quality.ParentRunID = parent.ID
			u.Quality.OutputHash = documentUnitOutputHash(*u)
			u.Quality.EntityVersions = nil
			for j := range u.Quality.Reviews {
				for k, id := range u.Quality.Reviews[j].InputIDs {
					u.Quality.Reviews[j].InputIDs[k] = ids[id]
				}
			}
		}
	}
	child.Steps = []workflows.StepState{{Status: "completed"}, {Status: "pending"}}
	documentQualityCounters(&child, units)
	return child, s.db.CreateDocumentRun(&child, units)
}
