package orchestrator

import (
	"errors"
	"strings"
	"testing"
)

func TestComposeNetworkPoolErrorHasRecovery(t *testing.T) {
	original := errors.New("failed to create network: all predefined address pools have been fully subnetted")
	err := explainComposeStartError("worker", original)
	for _, want := range []string{"worker", "주소 대역 소진", "docker network prune", "모델 가중치나 메모리 부족 문제가 아닙니다"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("missing %q: %v", want, err)
		}
	}
	if !errors.Is(err, original) {
		t.Fatal("original error lost")
	}
	other := errors.New("permission denied")
	if explainComposeStartError("local", other) != other || explainComposeStartError("local", nil) != nil {
		t.Fatal("unrelated errors changed")
	}
}
