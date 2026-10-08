package skills

import (
	"embed"
	"fmt"
	"sort"
	"strings"
)

//go:embed assets/*.md
var assets embed.FS

type Skill struct {
	Enabled      bool     `json:"enabled"`
	Builtin      bool     `json:"builtin"`
	Name         string   `json:"name"`
	Description  string   `json:"description"`
	Toolsets     []string `json:"toolsets"`
	Instructions string   `json:"instructions,omitempty"`
	asset        string
}

var builtins = []Skill{
	{Name: "task-planning", Description: "작업의 목적·입력·제약과 완료 조건을 정리합니다.", Toolsets: []string{}, asset: "assets/task-planning.md"},
	{Name: "code-writing", Description: "요구사항과 기존 구조에 맞춰 코드를 작성합니다.", Toolsets: []string{}, asset: "assets/code-writing.md"},
	{Name: "bug-diagnosis", Description: "재현과 관측 근거로 버그 원인을 좁힙니다.", Toolsets: []string{}, asset: "assets/bug-diagnosis.md"},
	{Name: "code-review", Description: "코드의 결함·누락·호환성을 검토합니다.", Toolsets: []string{}, asset: "assets/code-review.md"},
	{Name: "test-design", Description: "정상·경계·실패 사례를 설계하고 검증 결과를 구분합니다.", Toolsets: []string{}, asset: "assets/test-design.md"},
	{Name: "incident-analysis", Description: "서버의 관측 결과로 원인과 대응 순서를 정리합니다.", Toolsets: []string{}, asset: "assets/incident-analysis.md"},
	{Name: "document-writing", Description: "목적과 독자에 맞는 문서를 작성합니다.", Toolsets: []string{}, asset: "assets/document-writing.md"},
	{Name: "evidence-review", Description: "자료와 결과의 근거·시점·누락을 검토합니다.", Toolsets: []string{}, asset: "assets/evidence-review.md"},

	{Name: "web-research", Description: "여러 출처를 검색·검증해 근거 링크와 함께 답합니다.", Toolsets: []string{"web"}, asset: "assets/web-research.md"},
	{Name: "media-analysis", Description: "URL 영상·음성을 가져와 화면과 전사 내용을 구분해 분석합니다.", Toolsets: []string{"media"}, asset: "assets/media-analysis.md"},
	{Name: "image-creation", Description: "이미지 생성·편집 요청을 명확한 프롬프트와 설정으로 실행합니다.", Toolsets: []string{"image"}, asset: "assets/image-creation.md"},
	{Name: "ssh-inspection", Description: "등록된 SSH 서버를 최소한의 읽기 명령부터 안전하게 점검합니다.", Toolsets: []string{"ssh"}, asset: "assets/ssh-inspection.md"},
}

func Catalog() []Skill {
	items := make([]Skill, len(builtins))
	copy(items, builtins)
	for i := range items {
		items[i].Enabled = true
		items[i].Builtin = true
	}
	for index := range items {
		items[index].asset = ""
		items[index].Instructions = ""
	}
	sort.Slice(items, func(i, j int) bool { return items[i].Name < items[j].Name })
	return items
}

func Available(activeToolsets map[string]bool) []Skill {
	items := make([]Skill, 0, len(builtins))
	for _, item := range builtins {
		available := true
		for _, toolset := range item.Toolsets {
			if !activeToolsets[toolset] {
				available = false
				break
			}
		}
		if available {
			item.Enabled = true
			item.Builtin = true
			items = append(items, item)
		}
	}
	sort.Slice(items, func(i, j int) bool { return items[i].Name < items[j].Name })
	return items
}

func Load(name string, activeToolsets map[string]bool) (Skill, error) {
	name = strings.TrimSpace(name)
	for _, item := range Available(activeToolsets) {
		if item.Name != name {
			continue
		}
		data, err := assets.ReadFile(item.asset)
		if err != nil {
			return Skill{}, err
		}
		item.Instructions = strings.TrimSpace(string(data))
		return item, nil
	}
	return Skill{}, fmt.Errorf("skill is not available: %s", name)
}
