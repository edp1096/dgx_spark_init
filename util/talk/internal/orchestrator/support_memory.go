package orchestrator

import (
	"fmt"
	"gopkg.in/yaml.v3"
	"math"
	"strconv"
	"strings"
)

// CPU workers have bounded container memory. Their idle RSS is a startup
// allowance, not the peak of a document/browser/ffmpeg request. Only the active
// worker reserves its limit; idle peers are already included in MemAvailable.
func supportRequestMemoryGiB(c Component) (float64, error) {
	switch c.ComposeAsset {
	case "compose.extra-documents.yaml", "compose.extra-media.yaml", "compose.extra-collector.yaml":
	default:
		return 0, nil
	}
	data, err := assets.ReadFile("assets/" + c.ComposeAsset)
	if err != nil {
		return 0, err
	}
	var config struct {
		Services map[string]struct {
			Limit any `yaml:"mem_limit"`
		} `yaml:"services"`
	}
	if err = yaml.Unmarshal(data, &config); err != nil {
		return 0, err
	}
	s := strings.ToLower(strings.TrimSpace(fmt.Sprint(config.Services["runtime"].Limit)))
	multiplier := 1.0
	for _, u := range []struct {
		s string
		n float64
	}{{"g", 1 << 30}, {"m", 1 << 20}, {"k", 1 << 10}} {
		if strings.HasSuffix(s, u.s) {
			s = strings.TrimSuffix(s, u.s)
			multiplier = u.n
			break
		}
	}
	n, err := strconv.ParseFloat(s, 64)
	if err != nil || n <= 0 || math.IsNaN(n) || math.IsInf(n, 0) {
		return 0, fmt.Errorf("%s 작업 메모리 한도를 확인할 수 없습니다", c.Name)
	}
	return n * multiplier / (1 << 30), nil
}
