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
	Name         string   `json:"name"`
	Description  string   `json:"description"`
	Toolsets     []string `json:"toolsets"`
	Instructions string   `json:"instructions,omitempty"`
	asset        string
}

var builtins = []Skill{
	{Name: "web-research", Description: "여러 출처를 검색·검증해 근거 링크와 함께 답합니다.", Toolsets: []string{"web"}, asset: "assets/web-research.md"},
	{Name: "media-analysis", Description: "URL 영상·음성을 가져와 화면과 전사 내용을 구분해 분석합니다.", Toolsets: []string{"media"}, asset: "assets/media-analysis.md"},
	{Name: "image-creation", Description: "이미지 생성·편집 요청을 명확한 프롬프트와 설정으로 실행합니다.", Toolsets: []string{"image"}, asset: "assets/image-creation.md"},
	{Name: "ssh-inspection", Description: "등록된 SSH 서버를 최소한의 읽기 명령부터 안전하게 점검합니다.", Toolsets: []string{"ssh"}, asset: "assets/ssh-inspection.md"},
}

func Catalog() []Skill {
	items := make([]Skill, len(builtins))
	copy(items, builtins)
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
