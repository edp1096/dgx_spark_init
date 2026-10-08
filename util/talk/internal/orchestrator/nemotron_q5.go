package orchestrator

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
)

const (
	nemotronQ5File    = "nemotron-3.5-asr-streaming-0.6b.q5_k.gguf"
	nemotronQ5SHA256  = "f0dab30ca22a606c2ab9a65efae88421851c6ca4ab01160708ab733b4467ce58"
	nemotronF16File   = "nemotron-3.5-asr-streaming-0.6b.f16.gguf"
	nemotronF16SHA256 = "cc5fc1b6e05fdb905c4d9bc01842e1d297650cc39e117334bf16d64eb2b55fc0"
)

func hostModelMatches(ctx context.Context, host Host, path, expected string) bool {
	out, err := executeHost(ctx, host, nil, "sha256sum", path)
	fields := strings.Fields(string(out))
	return err == nil && len(fields) > 0 && fields[0] == expected
}

func nemotronSourceAsset() modelAsset {
	const file = "nemotron-3.5-asr-streaming-0.6b.nemo"
	return modelAsset{Repo: "nvidia/nemotron-3.5-asr-streaming-0.6b", Revision: "ea30d66debe3740a08b573244286791d423d6b3e", Path: "/tmp/models", Files: []string{file}, SHA256: map[string]string{file: "210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74"}}
}

// The Q5 checkpoint is derived locally, never requested under a nonexistent
// upstream filename. An existing qualified output requires no conversion.
func (c *Controller) prepareNemotronQ5Weights(ctx context.Context, component Component, host Host, data, modelDir string) error {
	if hostModelMatches(ctx, host, filepath.Join(modelDir, nemotronQ5File), nemotronQ5SHA256) {
		return nil
	}
	memory, err := c.hostMemory(ctx, component.Host)
	if err != nil {
		return err
	}
	if memory.AvailableGiB < 14 {
		return fmt.Errorf("Nemotron Q5_K 최초 CPU 변환에는 14 GiB 여유가 필요합니다. 언어 모델 기동 전에 ASR 모델 준비를 실행하세요")
	}
	build := filepath.Join(data, "runtime", "nemotron-model-converter")
	if err = materializeBuildAssets(ctx, host, "nemotron-asr", build); err != nil {
		return err
	}
	const image = "sparktalk-nemotron-converter:6a3ca369370782acee0dd155e34394ba60b920c4"
	if _, err = executeHost(ctx, host, nil, "docker", "image", "inspect", image); err != nil {
		if _, err = executeHost(ctx, host, nil, "docker", "build", "--target", "converter", "-t", image, build); err != nil {
			return err
		}
	}
	user, err := executionUser(ctx, host)
	if err != nil {
		return err
	}
	out, err := executeHost(ctx, host, nil, "docker", "run", "--rm", "--network", "none", "--user", user,
		"-e", "HOME=/tmp", "--memory", "12g", "--memory-swap", "12g", "-v", modelDir+":/models",
		"--entrypoint", "python", image, "/opt/nemo-quant/prepare_q5.py")
	if report := recipeReporter(ctx); report != nil {
		report(string(out))
	}
	if err != nil {
		return fmt.Errorf("Nemotron Q5_K 변환 실패: %w: %s", err, out)
	}
	if !hostModelMatches(ctx, host, filepath.Join(modelDir, nemotronQ5File), nemotronQ5SHA256) {
		return fmt.Errorf("Nemotron Q5_K 준비 후 SHA-256 검증 실패")
	}
	return nil
}
