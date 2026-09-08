package orchestrator

import (
	"testing"
	"time"
)

func TestOperationKeepsStableStageHistory(t *testing.T) {
	controller, err := NewController()
	if err != nil {
		t.Fatal(err)
	}
	if err := controller.begin(Operation{Action: "start", State: "running", Phase: "기동 계획 준비", StartedAt: time.Now()}); err != nil {
		t.Fatal(err)
	}
	controller.updateOperation("flash-next", progressInfo{Key: "main-weights", Phase: "체크포인트 적재", Detail: "1/206", Progress: .2})
	controller.updateOperation("flash-next", progressInfo{Key: "main-weights", Phase: "체크포인트 적재", Detail: "118/206", Progress: .5})
	controller.updateOperation("flash-next", progressInfo{Key: "kv-cache", Phase: "KV 캐시 할당", Progress: .89})

	controller.mu.RLock()
	op := controller.op
	controller.mu.RUnlock()
	if len(op.Steps) != 3 {
		t.Fatalf("expected plan and two stable stages, got %#v", op.Steps)
	}
	if op.Steps[1].Detail != "118/206" || op.Steps[1].State != "complete" {
		t.Fatalf("checkpoint stage was not updated in place: %#v", op.Steps[1])
	}
	if op.Steps[2].State != "current" {
		t.Fatalf("latest stage should be current: %#v", op.Steps[2])
	}

	controller.finishOperation("complete", "")
	if controller.op.Steps[2].State != "complete" || controller.op.Phase != "준비 완료" {
		t.Fatalf("operation did not finish its last stage: %#v", controller.op)
	}
}
