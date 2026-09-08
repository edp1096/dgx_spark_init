package workflows

import "sparktalk/internal/skills"

type Step struct {
	Name          string   `json:"name"`
	Skills        []string `json:"skills"`
	Goal          string   `json:"goal"`
	DoneWhen      string   `json:"done_when"`
	VerifyTool    string   `json:"verify_tool,omitempty"`
	VerifyCommand string   `json:"verify_command,omitempty"`
	OnFailure     int      `json:"on_failure"` // -1 stops; otherwise return to an earlier step.
	MaxRetries    int      `json:"max_retries"`
}
type Definition struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Enabled     bool   `json:"enabled"`
	Builtin     bool   `json:"builtin"`
	Steps       []Step `json:"steps"`
}
type Evidence struct {
	ID        string `json:"id"`
	Tool      string `json:"tool"`
	Arguments string `json:"arguments"`
	Result    string `json:"result"`
	Error     string `json:"error,omitempty"`
}
type Report struct {
	Handoff  string   `json:"handoff,omitempty"`
	Status   string   `json:"status"`
	Summary  string   `json:"summary"`
	Evidence []string `json:"evidence"`
}
type Attempt struct {
	Status   string     `json:"status"`
	Summary  string     `json:"summary"`
	Error    string     `json:"error,omitempty"`
	Evidence []Evidence `json:"evidence"`
}
type StepState struct {
	Handoff  string     `json:"handoff,omitempty"`
	History  []Attempt  `json:"history,omitempty"`
	Status   string     `json:"status"`
	Summary  string     `json:"summary"`
	Error    string     `json:"error,omitempty"`
	Evidence []Evidence `json:"evidence"`
	Attempts int        `json:"attempts"`
	Retries  int        `json:"retries"`
}
type Run struct {
	ID         string                  `json:"id"`
	SessionID  string                  `json:"session_id"`
	Status     string                  `json:"status"`
	Goal       string                  `json:"goal"`
	Definition Definition              `json:"definition"`
	Skills     map[string]skills.Skill `json:"skills,omitempty"`
	Steps      []StepState             `json:"steps"`
	Current    int                     `json:"current"`
	Updated    string                  `json:"updated"`
}

func Defaults() []Definition {
	step := func(name, skill, goal, done, tool string) Step {
		return Step{Name: name, Skills: []string{skill}, Goal: goal, DoneWhen: done, VerifyTool: tool, OnFailure: -1}
	}
	defs := []Definition{
		{Name: "code-development", Description: "요구사항을 정리하고 코드를 작성한 뒤 리뷰합니다. 실행 검증은 별도 단계로 추가할 수 있습니다.", Steps: []Step{step("요구사항", "task-planning", "요구사항과 구현 범위를 정리한다.", "입력·출력·제약과 검증 방법이 명확하다.", ""), step("코드 작성", "code-writing", "앞 단계의 요구사항에 맞는 코드를 작성한다.", "필요한 코드와 사용법을 제공한다. 실행 여부를 구분한다.", ""), step("코드 리뷰", "code-review", "앞 단계 코드를 검토하고 발견한 문제를 반영한 최종 결과를 작성한다.", "코드·변경 내용·검증 여부와 남은 한계를 제시한다.", "")}},
		{Name: "bug-repair", Description: "등록된 서버에서 원인을 진단하고 수정·검증합니다. SSH 권한이 필요합니다.", Steps: []Step{step("원인 진단", "bug-diagnosis", "실제 코드와 오류를 조사하고 원인을 좁힌다.", "문제 재현 또는 진단 근거를 확보한다.", "ssh_exec"), step("수정", "code-writing", "확인된 원인을 수정한다.", "수정 파일과 변경 내용을 확인한다.", "ssh_exec"), step("검증", "test-design", "수정한 동작의 테스트를 실제로 실행한다.", "관련 테스트 명령의 성공 결과를 확보한다.", "ssh_exec")}},
		{Name: "server-diagnosis", Description: "서버 상태·로그를 확인하고 원인과 대응 방안을 정리합니다.", Steps: []Step{step("상태 확인", "ssh-inspection", "요청한 서버의 상태와 관련 로그를 확인한다.", "실제 조회 결과를 확보한다.", "ssh_exec"), step("진단 보고", "incident-analysis", "관측 결과로 원인을 분석하고 조치 및 검증 방법을 제시한다.", "사실·가설·미확인을 구분하고 대응 순서를 제시한다.", "")}},
		{Name: "research-report", Description: "자료 조사 → 근거 대조 → 보고서 작성", Steps: []Step{step("자료 조사", "web-research", "질문에 필요한 자료와 원출처를 조사한다.", "관련 자료와 출처 링크를 확보한다.", "web_fetch"), step("근거 대조", "evidence-review", "조사 결과의 출처·시점·상충 내용을 검토한다.", "사실과 추정, 확인하지 못한 내용을 구분한다.", ""), step("보고서", "document-writing", "검증 결과를 바탕으로 질문에 답하는 보고서를 작성한다.", "결론과 근거 링크, 한계를 간결하게 제시한다.", "")}},
		{Name: "media-report", Description: "영상·음성을 가져와 분석하고 근거를 확인해 요약합니다.", Steps: []Step{step("미디어 분석", "media-analysis", "사용자가 제공한 미디어를 가져와 화면과 음성을 분석한다.", "실제 미디어를 확보하고 전사 여부를 명시한다.", "media_import"), step("분석 요약", "evidence-review", "화면·전사·추정을 구분해서 최종 분석을 작성한다.", "확인한 장면과 음성에 근거해 요약한다.", "")}},
		{Name: "document-production", Description: "자료와 목적을 정리하고 문서를 작성·검토합니다.", Steps: []Step{step("구성", "task-planning", "독자와 목적, 제공된 자료를 바탕으로 문서 구성을 정한다.", "목적·범위·구성이 명확하다.", ""), step("작성", "document-writing", "구성에 맞춰 문서 본문을 작성한다.", "바로 사용할 수 있는 초안을 제공한다.", ""), step("검토", "evidence-review", "초안의 사실·누락·표현을 검토해 최종 문서를 작성한다.", "최종 문서와 확인이 필요한 사항을 구분한다.", "")}},
		{Name: "image-production", Description: "제작 요구사항을 정리하고 이미지를 생성한 뒤 결과를 확인합니다.", Steps: []Step{step("제작 구상", "task-planning", "이미지의 목적·내용·구도·스타일을 정리한다.", "생성에 사용할 구체적인 요구사항이 준비됐다.", ""), step("이미지 제작", "image-creation", "앞 단계 요구사항에 맞게 이미지를 생성한다.", "실제 생성 결과를 확보하고 결과와 한계를 안내한다.", "image_generate")}},
	}
	defs[1].Steps[2].OnFailure = 1
	defs[1].Steps[2].MaxRetries = 1
	for i := range defs {
		defs[i].Enabled = true
		defs[i].Builtin = true
	}
	return defs
}
