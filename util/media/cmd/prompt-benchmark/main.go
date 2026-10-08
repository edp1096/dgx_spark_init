package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
	"unicode/utf8"

	mediaprompt "mediaapp/internal/prompt"
)

type testCase struct {
	ID        string   `json:"id"`
	Category  string   `json:"category"`
	Mode      string   `json:"mode"`
	System    string   `json:"-"`
	Input     string   `json:"input"`
	MinWords  int      `json:"min_words,omitempty"`
	MaxWords  int      `json:"max_words,omitempty"`
	Must      []string `json:"must,omitempty"`
	Exact     []string `json:"exact,omitempty"`
	Forbidden []string `json:"forbidden,omitempty"`
}

type check struct {
	Name   string `json:"name"`
	Passed bool   `json:"passed"`
	Detail string `json:"detail,omitempty"`
}

type result struct {
	Test       testCase `json:"test"`
	Output     string   `json:"output"`
	Words      int      `json:"words"`
	LatencyMS  int64    `json:"latency_ms"`
	Checks     []check  `json:"checks"`
	Passed     int      `json:"passed"`
	Total      int      `json:"total"`
	Error      string   `json:"error,omitempty"`
	Usage      any      `json:"usage,omitempty"`
	APITimings any      `json:"api_timings,omitempty"`
}

type report struct {
	Engine      string    `json:"engine"`
	Endpoint    string    `json:"endpoint"`
	Model       string    `json:"model"`
	GeneratedAt time.Time `json:"generated_at"`
	Results     []result  `json:"results"`
	Passed      int       `json:"passed"`
	Total       int       `json:"total"`
	MeanMS      float64   `json:"mean_latency_ms"`
}

func suite() []testCase {
	return []testCase{
		{ID: "t2i-short-en", Category: "short expansion", Mode: "t2i", Input: "Student girl is dancing.", MinWords: 55, MaxWords: 100, Must: []string{"student girl", "danc"}},
		{ID: "t2i-short-ko", Category: "Korean translation", Mode: "t2i", Input: "비 오는 밤의 서울 골목에서 빨간 우산을 든 젊은 한국인 여성", MinWords: 55, MaxWords: 100, Must: []string{"rain", "Seoul", "red umbrella", "Korean woman"}},
		{ID: "t2i-visible-text-en", Category: "exact visible text", Mode: "t2i", Input: "A studio product photograph of a matte black wristwatch. The watch face must show exactly \"SPARK 07:30\". No logo and no other text.", Must: []string{"matte black", "wristwatch", "no logo", "no other text"}, Exact: []string{"\"SPARK 07:30\""}},
		{ID: "t2i-visible-text-ko", Category: "exact Korean text", Mode: "t2i", Input: "밤의 작은 네온 간판 가게. 간판에는 정확히 \"오늘도 빛나\"라고 쓰고, 다른 글자는 절대 넣지 않는다.", Must: []string{"neon", "no other text"}, Exact: []string{"\"오늘도 빛나\""}},
		{ID: "t2i-count-color-space", Category: "count and spatial binding", Mode: "t2i", Input: "Exactly three ceramic cups in one row: a small red cup on the left, a tall cobalt-blue cup in the center, and a wide cream cup on the right. Nothing else on the table.", Must: []string{"exactly three", "red", "left", "cobalt-blue", "center", "cream", "right", "nothing else"}},
		{ID: "t2i-multi-subject-binding", Category: "multi-subject binding", Mode: "t2i", Input: "On the left, an elderly woman in a yellow raincoat holds a silver cane. On the right, a teenage boy in a green cap carries a blue backpack. Neither person holds an umbrella.", Must: []string{"left", "elderly woman", "yellow raincoat", "silver cane", "right", "teenage boy", "green cap", "blue backpack", "Neither", "umbrella"}},
		{ID: "t2i-negation", Category: "negative constraints", Mode: "t2i", Input: "An empty white gallery with one blue cube in the exact center. No people, no windows, no plants, no signs, and no visible text.", Must: []string{"empty white gallery", "one blue cube", "center", "no people", "no windows", "no plants", "no signs", "no visible text"}},
		{ID: "t2i-detailed-preserve", Category: "detailed prompt restraint", Mode: "t2i", Input: "A documentary-style 35mm color photograph, eye-level medium shot, of two female marine biologists in orange dry suits kneeling beside a tide pool at dawn; the older scientist on the left labels a glass vial while the younger scientist on the right photographs a purple starfish without touching it. Cool fog, wet basalt, subdued cyan and amber palette, natural grain, no cinematic glow, no extra people.", MaxWords: 120, Must: []string{"documentary-style", "35mm", "two female marine biologists", "orange dry suits", "older scientist", "left", "glass vial", "younger scientist", "right", "purple starfish", "without touching", "no cinematic glow", "no extra people"}},
		{ID: "t2i-composer-restraint", Category: "composer consolidation", Mode: "t2i", Input: "subject: middle-aged Korean potter; action: trimming a celadon bowl; wardrobe: indigo work apron; setting: sunlit rural workshop; framing: waist-up three-quarter view; lighting: soft north-window light; texture: clay dust on hands; palette: celadon green, warm wood, indigo; style: editorial photograph; constraints: one person, no text, no extra pottery tools", MaxWords: 115, Must: []string{"middle-aged Korean potter", "celadon bowl", "indigo work apron", "rural workshop", "three-quarter", "north-window light", "clay dust", "one person", "no text", "no extra pottery tools"}},
		{ID: "t2i-unusual-medium", Category: "medium preservation", Mode: "t2i", Input: "A childlike wax-crayon drawing on rough beige paper of a purple rhinoceros riding a yellow bicycle, front view, intentionally uneven outlines, only five colors, no photorealism.", Must: []string{"wax-crayon", "rough beige paper", "purple rhinoceros", "yellow bicycle", "front view", "uneven outlines", "five colors", "no photorealism"}, Forbidden: []string{"photorealistic", "cinematic photograph"}},

		{ID: "control-minimal", Category: "control non-invention", Mode: "control", Input: "An elderly Korean woman wearing an indigo hanbok, natural skin texture, soft overcast light.", MinWords: 40, MaxWords: 80, Must: []string{"elderly Korean woman", "indigo hanbok", "natural skin texture", "soft overcast light"}, Forbidden: []string{"standing", "sitting", "left", "right", "low-angle", "high-angle", "close-up", "full-body"}},
		{ID: "control-text", Category: "control exact text", Mode: "control", Input: "A courier wearing a plain gray jacket with the exact chest patch text \"NORTH-12\". Preserve the reference geometry; add no bag and no hat.", Must: []string{"gray jacket", "no bag", "no hat"}, Exact: []string{"\"NORTH-12\""}},
		{ID: "control-count", Category: "control count binding", Mode: "control", Input: "Exactly two red wooden mannequins and one blue metal mannequin, no additional figures, matte materials, clean studio lighting.", Must: []string{"exactly two", "red wooden", "one blue metal", "no additional figures", "matte", "studio lighting"}},
		{ID: "control-style-only", Category: "control style restraint", Mode: "control", Input: "Render the referenced structure as a monochrome charcoal sketch on ivory paper, preserving its composition and silhouette, with no added background objects.", Must: []string{"monochrome charcoal sketch", "ivory paper", "preserving", "composition", "silhouette", "no added background objects"}, Forbidden: []string{"camera", "angle"}},

		{ID: "edit-clothing-preserve", Category: "edit preservation", Mode: "edit", Input: "Change: replace only the blue jacket with a red raincoat. Preserve: the same person's identity, face, hairstyle, pose, hands, framing, cafe background, warm lighting, and every other garment.", MinWords: 45, MaxWords: 100, Must: []string{"blue jacket", "red raincoat", "identity", "face", "hairstyle", "pose", "hands", "framing", "cafe background", "warm lighting", "every other garment"}},
		{ID: "edit-expression", Category: "minimal edit", Mode: "edit", Input: "Change: make the subject wink with the left eye only. Preserve: identity, right eye, mouth expression, head angle, hair, clothing, background, lighting, crop, and image style.", MinWords: 45, MaxWords: 100, Must: []string{"wink", "left eye", "right eye", "mouth expression", "head angle", "identity", "background", "lighting", "crop"}},
		{ID: "edit-multi-reference", Category: "reference role binding", Mode: "edit", Input: "Change: dress the person from the primary image in the black lace bodysuit from the clothing reference and match the body pose from the pose reference. Preserve: primary person's face, identity, hair color, body proportions, room, camera angle, and lighting.", Must: []string{"primary image", "black lace bodysuit", "clothing reference", "pose reference", "face", "identity", "hair color", "body proportions", "room", "camera angle", "lighting"}},
		{ID: "edit-sign-text", Category: "edit exact text", Mode: "edit", Input: "Change: replace only the cafe sign wording with exactly \"달빛 다방\". Preserve: sign shape, letter placement, all people, faces, tables, reflections, framing, color grade, and all other visible text unchanged.", Must: []string{"replace only", "all other visible text unchanged"}, Exact: []string{"\"달빛 다방\""}},

		{ID: "paint-inpaint", Category: "masked replacement", Mode: "paint", Input: "Inside the painted mask only, replace the broken lamp with a small brass desk lamp matching the room's warm light and perspective. Keep everything outside the mask unchanged.", Must: []string{"mask", "broken lamp", "small brass desk lamp", "warm light", "perspective", "outside", "unchanged"}},
		{ID: "paint-remove", Category: "masked removal", Mode: "paint", Input: "Remove the person inside the mask and reconstruct the uninterrupted brick wall and pavement behind them. Do not alter any unmasked person, shadow, sign, or window.", Must: []string{"remove", "person", "brick wall", "pavement", "unmasked", "shadow", "sign", "window"}},
		{ID: "paint-outpaint-no-prompt-drift", Category: "outpaint continuity", Mode: "paint", Input: "Extend the image to the right with a seamless continuation of the existing beach, horizon, sky, lighting, lens perspective, and film grain. Introduce no new people, buildings, boats, animals, or text.", Must: []string{"right", "seamless", "beach", "horizon", "sky", "lighting", "lens perspective", "film grain", "no new people", "buildings", "boats", "animals", "text"}},
		{ID: "paint-text", Category: "masked exact text", Mode: "paint", Input: "Replace only the masked label with clean black lettering that reads exactly \"BATCH 042\" while matching the paper texture and print perspective. All unmasked packaging remains identical.", Must: []string{"masked label", "black lettering", "paper texture", "print perspective", "unmasked packaging", "identical"}, Exact: []string{"\"BATCH 042\""}},

		{ID: "t2v-chronology", Category: "video chronology", Mode: "t2v", Input: "A red paper airplane rests on a school desk. A breeze lifts it, it circles once around the empty classroom, then lands on the same desk. Static eye-level camera. Only soft wind and paper rustling; no music and no people.", MinWords: 150, MaxWords: 220, Must: []string{"red paper airplane", "school desk", "breeze", "circles once", "empty classroom", "same desk", "static", "eye-level", "wind", "paper rustling", "no music", "no people"}},
		{ID: "t2v-dialogue", Category: "exact dialogue", Mode: "t2v", Input: "At a quiet Seoul bus stop at night, an elderly Korean man checks the empty road, smiles, and says exactly in Korean: \"오늘은 천천히 가도 돼.\" A bus approaches but does not arrive before the shot ends. Slow dolly-in, medium shot. Rain ambience only, no music.", MinWords: 150, MaxWords: 220, Must: []string{"Seoul bus stop", "elderly Korean man", "empty road", "bus approaches", "does not arrive", "slow dolly-in", "medium shot", "rain", "no music"}, Exact: []string{"오늘은 천천히 가도 돼."}},
		{ID: "t2v-count-direction", Category: "video count and direction", Mode: "t2v", Input: "Exactly four white robots walk from right to left across a red salt flat in one continuous line. The second robot briefly waves once, but the others never raise their arms. Wide locked-off shot, no cuts, no vehicles, only footsteps and wind.", MinWords: 150, MaxWords: 220, Must: []string{"exactly four", "white robots", "right to left", "red salt flat", "second robot", "waves once", "others", "never raise", "wide", "locked-off", "no cuts", "no vehicles", "footsteps", "wind"}},
		{ID: "t2v-no-invention", Category: "video negative constraints", Mode: "t2v", Input: "A single candle burns steadily in a completely dark room for the entire shot. Extreme close-up, camera remains static. The flame does not flicker or go out. No hands, faces, windows, smoke, dialogue, sound effects, or music; only near-silence.", MinWords: 150, MaxWords: 220, Must: []string{"single candle", "entire shot", "extreme close-up", "static", "does not flicker", "go out", "no hands", "faces", "windows", "smoke", "dialogue", "sound effects", "music", "near-silence"}},

		{ID: "sub-ko-marker-batch", Category: "subtitle marker integrity", Mode: "subtitle", System: koreanBatchSystem, Input: "[[0000]] すみません、次の電車は何時ですか？\n[[0001]] The last train left ten minutes ago.\n[[0002]] じゃあ、今夜はここに泊まるしかないね。\n[[0003]] Don't jump to conclusions. I'll call a taxi.\n[[0004]] 本当に助かるよ。\n[[0005]] It should be here by 12:45.\n[[0006]] 料金は三千円くらいかな。\n[[0007]] Probably, unless traffic gets worse.", Exact: []string{"[[0000]]", "[[0001]]", "[[0002]]", "[[0003]]", "[[0004]]", "[[0005]]", "[[0006]]", "[[0007]]", "12:45"}, Must: []string{"막차", "택시", "교통", "천"}},
		{ID: "sub-ko-idioms", Category: "subtitle idiom translation", Mode: "subtitle", System: koreanBatchSystem, Input: "[[0000]] We're not out of the woods yet.\n[[0001]] You really hit the nail on the head.\n[[0002]] Let's call it a day before this gets out of hand.", Exact: []string{"[[0000]]", "[[0001]]", "[[0002]]"}, Must: []string{"아직", "정확", "오늘"}, Forbidden: []string{"woods", "nail", "call it a day"}},
		{ID: "sub-ko-register", Category: "subtitle register and honorifics", Mode: "subtitle", System: koreanBatchSystem, Input: "[[0000]] 先生、昨日はご迷惑をおかけして申し訳ありませんでした。\n[[0001]] 気にしなくていいよ。次から気をつけてね。\n[[0002]] はい、必ず確認いたします。", Exact: []string{"[[0000]]", "[[0001]]", "[[0002]]"}, Must: []string{"선생님", "죄송", "다음부터", "확인"}},
		{ID: "sub-ko-technical", Category: "subtitle technical accuracy", Mode: "subtitle", System: koreanBatchSystem, Input: "[[0000]] Set the aperture to f/2.8 and keep the shutter at one-fiftieth.\n[[0001]] The encoder is dropping frames because the buffer is full.\n[[0002]] Do not normalize the dialogue track before removing the noise floor.", Exact: []string{"[[0000]]", "[[0001]]", "[[0002]]", "f/2.8", "1/50"}, Must: []string{"조리개", "셔터", "인코더", "프레임", "버퍼", "노이즈"}},
		{ID: "sub-ko-names-numbers", Category: "subtitle entity preservation", Mode: "subtitle", System: koreanBatchSystem, Input: "[[0000]] Dr. Reyes sent build v2.7.14 to Mina at 09:05.\n[[0001]] Keep API_KEY unchanged and restart node GB10-A.\n[[0002]] The checksum is A7F3-19C0, not A7F3-91C0.", Exact: []string{"[[0000]]", "[[0001]]", "[[0002]]", "v2.7.14", "09:05", "API_KEY", "GB10-A", "A7F3-19C0", "A7F3-91C0"}, Must: []string{"미나"}},
		{ID: "sub-en-from-ko", Category: "Korean to English subtitle", Mode: "subtitle", System: englishBatchSystem, Input: "[[0000]] 오늘 회의는 취소된 게 아니라 내일 오전으로 미뤄졌습니다.\n[[0001]] 비가 그치면 장비부터 옮기자.\n[[0002]] 설마 그 말을 진심으로 한 건 아니지?", Exact: []string{"[[0000]]", "[[0001]]", "[[0002]]"}, Must: []string{"postponed", "tomorrow morning", "rain", "equipment", "serious"}},
		{ID: "sub-en-from-ja", Category: "Japanese ambiguity to English", Mode: "subtitle", System: englishBatchSystem, Input: "[[0000]] ちょっと、それは聞いてないよ。\n[[0001]] まあ、悪くはないんじゃない？\n[[0002]] 先に行ってて。すぐ追いつくから。", Exact: []string{"[[0000]]", "[[0001]]", "[[0002]]"}, Must: []string{"didn't", "not bad", "go ahead", "catch up"}},
	}
}

const koreanBatchSystem = "당신은 전문 영상 자막 번역가입니다. 각 [[NNNN]] 표식을 그대로 유지하면서 뒤의 자막을 자연스러운 한국어로 번역하세요. 일본어·영어 원문을 복사하지 말고 설명 없이 번역문만 출력하세요."

const englishBatchSystem = "You translate subtitle segments. Translate only the text into English. Preserve every [[NNNN]] marker exactly once and in order. Do not add explanations."

func main() {
	engine := flag.String("engine", "engine", "engine name recorded in the report")
	endpoint := flag.String("endpoint", "http://127.0.0.1:8696", "OpenAI-compatible API base URL")
	model := flag.String("model", "huihui-gemma4-e2b", "model or alias")
	output := flag.String("output", "prompt-benchmark.json", "JSON report path")
	warmup := flag.Bool("warmup", true, "run one unrecorded warm-up request")
	flag.Parse()

	client := &http.Client{Timeout: 3 * time.Minute}
	if *warmup {
		_, _ = run(client, *endpoint, *model, testCase{Mode: "t2i", Input: "A red apple on a white plate."})
	}

	rep := report{Engine: *engine, Endpoint: *endpoint, Model: *model, GeneratedAt: time.Now()}
	var latency int64
	for _, tc := range suite() {
		r, err := run(client, *endpoint, *model, tc)
		if err != nil {
			r.Error = err.Error()
		}
		r.Checks = evaluate(tc, r.Output)
		for _, c := range r.Checks {
			r.Total++
			rep.Total++
			if c.Passed {
				r.Passed++
				rep.Passed++
			}
		}
		latency += r.LatencyMS
		rep.Results = append(rep.Results, r)
		fmt.Printf("%-28s %3d/%-3d %5d words %6d ms\n", tc.ID, r.Passed, r.Total, r.Words, r.LatencyMS)
	}
	if len(rep.Results) > 0 {
		rep.MeanMS = float64(latency) / float64(len(rep.Results))
	}
	data, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(*output, append(data, '\n'), 0o644); err != nil {
		panic(err)
	}
	fmt.Printf("TOTAL %d/%d (%.1f%%), mean %.0f ms -> %s\n", rep.Passed, rep.Total, 100*float64(rep.Passed)/float64(rep.Total), rep.MeanMS, *output)
}

func run(client *http.Client, endpoint, model string, tc testCase) (result, error) {
	system := tc.System
	if system == "" {
		system = mediaprompt.System(tc.Mode, false)
	}
	payload := map[string]any{
		"model": model,
		"messages": []map[string]any{
			{"role": "system", "content": system},
			{"role": "user", "content": userContent(tc)},
		},
		"max_completion_tokens": 600,
		"temperature":           0,
		"top_k":                 1,
		"seed":                  42,
		"reasoning_effort":      "none",
	}
	body, _ := json.Marshal(payload)
	started := time.Now()
	resp, err := client.Post(strings.TrimRight(endpoint, "/")+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	r := result{Test: tc, LatencyMS: time.Since(started).Milliseconds()}
	if err != nil {
		return r, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	r.LatencyMS = time.Since(started).Milliseconds()
	if err != nil {
		return r, err
	}
	if resp.StatusCode/100 != 2 {
		return r, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	var decoded struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage   any `json:"usage"`
		Timings any `json:"timings"`
	}
	if err := json.Unmarshal(data, &decoded); err != nil {
		return r, err
	}
	if len(decoded.Choices) == 0 {
		return r, fmt.Errorf("response has no choices")
	}
	r.Output = strings.TrimSpace(decoded.Choices[0].Message.Content)
	r.Words = len(strings.Fields(r.Output))
	r.Usage = decoded.Usage
	r.APITimings = decoded.Timings
	return r, nil
}

func userContent(tc testCase) string {
	if tc.System != "" {
		return tc.Input
	}
	return "User Raw Input Prompt: " + tc.Input
}

func evaluate(tc testCase, output string) []check {
	checks := []check{}
	add := func(name string, passed bool, detail string) {
		checks = append(checks, check{Name: name, Passed: passed, Detail: detail})
	}
	words := len(strings.Fields(output))
	if tc.MinWords > 0 {
		add("minimum word count", words >= tc.MinWords, fmt.Sprintf("%d >= %d", words, tc.MinWords))
	}
	if tc.MaxWords > 0 {
		add("maximum word count", words <= tc.MaxWords, fmt.Sprintf("%d <= %d", words, tc.MaxWords))
	}
	lower := strings.ToLower(output)
	for _, needle := range tc.Must {
		add("contains: "+needle, strings.Contains(lower, strings.ToLower(needle)), "")
	}
	for _, needle := range tc.Exact {
		add("exact: "+needle, strings.Contains(output, needle), "")
	}
	for _, needle := range tc.Forbidden {
		add("avoids: "+needle, !strings.Contains(lower, strings.ToLower(needle)), "")
	}
	add("single paragraph", !strings.Contains(output, "\n"), "")
	trimmed := strings.TrimSpace(output)
	badPrefix := strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[") || strings.HasPrefix(trimmed, "#") || strings.HasPrefix(trimmed, "-") || strings.HasPrefix(trimmed, "Output:") || strings.HasPrefix(trimmed, "Prompt:")
	add("no wrapper", !badPrefix, "")
	add("valid UTF-8", utf8.ValidString(output), "")
	return checks
}
