package orchestrator

import "encoding/json"

const QwenFP4Local = "edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local"

// This locally built checkpoint has no published Hub revision. Its immutable
// build manifest identifies the transferred weights independently of later
// runtime reports or display-name changes.
var qwenFP4LocalRelease = func() modelAsset {
	raw, err := assets.ReadFile("assets/model-prepare/fp4-local.json")
	if err != nil {
		panic(err)
	}
	var release modelAsset
	if err := json.Unmarshal(raw, &release); err != nil {
		panic(err)
	}
	if release.Repo != QwenFP4Local || release.Path != "/hf/"+QwenFP4Local || !release.LocalOnly || len(release.SHA256["qualification/build-transfer-manifest.json"]) != 64 {
		panic("invalid FP4-Local checkpoint identity")
	}
	return release
}()
