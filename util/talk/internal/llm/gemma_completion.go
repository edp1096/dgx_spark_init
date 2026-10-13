package llm

import "strings"

// Gemma31 can prioritize persona over the requested deliverable. Keep the task
// scope policy ahead of the long tool prefix, without replacing the persona.
const gemma31CompletionPolicy = "[답변·결과물 작성 원칙] 생각이나 추론을 짧게 하라는 지시는 최종 답변을 축약하라는 뜻이 아니다. 페르소나는 대화의 말투와 호칭에 적용하며, 설명의 정보량이나 작품의 완성도를 줄이는 근거로 삼지 않는다. 설명 요청에는 핵심 항목의 배경, 특징과 구체적인 사례를 충분히 풀어서 설명한다. 소설 요청에는 줄거리나 시놉시스 대신 장면 묘사, 인물의 행동과 대사가 이어지는 소설 본문을 쓴다. 도입·사건·반전·결말이라는 표제 아래 줄거리를 한 문단씩 요약하는 것으로 작품을 대체하지 않는다. 작품 본문의 서술 문체는 작품에 맞게 쓰고, 사용자 호칭과 어시스턴트의 대화 말투를 작품 속 서술에 억지로 끼워 넣지 않는다. 요청한 작업을 끝까지 완성하고 사용자가 지정한 범위와 분량을 지킨다. 사용자가 짧은 답변이나 요약을 요청했거나 간단한 질문이면 간결하게 답한다."

func (c *Client) withGemma31CompletionPolicy(messages []Message) []Message {
	if c.modelType != "gemma4" && c.modelType != "gemma4-vllm" {
		return messages
	}
	model := strings.ToLower(c.model)
	if !strings.Contains(model, "gemma-4-31b") && !strings.Contains(model, "gemma4-31b") {
		return messages
	}
	if len(messages) == 0 || messages[0].Role != "system" {
		return messages
	}
	prompt, ok := messages[0].Content.(string)
	if !ok || strings.Contains(prompt, gemma31CompletionPolicy) {
		return messages
	}
	result := append([]Message(nil), messages...)
	result[0].Content = gemma31CompletionPolicy + "\n\n" + prompt
	return result
}
