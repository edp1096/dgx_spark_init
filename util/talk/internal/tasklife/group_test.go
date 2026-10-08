package tasklife

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestStopCancelsWaitsAndRejects(t *testing.T) {
	var g Group
	ctx, finish, err := g.Track(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	g.Stop()
	if ctx.Err() != context.Canceled {
		t.Fatal("not canceled")
	}
	if _, _, err = g.Track(context.Background()); !errors.Is(err, ErrStopping) {
		t.Fatal(err)
	}
	limit, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if err = g.Wait(limit); err != context.DeadlineExceeded {
		t.Fatal("did not wait", err)
	}
	finish()
	finish()
	if err = g.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
}
func TestConcurrentAdmissionAndStop(t *testing.T) {
	for i := 0; i < 100; i++ {
		var g Group
		var wg sync.WaitGroup
		for j := 0; j < 20; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				ctx, done, err := g.Track(context.Background())
				if err == nil {
					<-ctx.Done()
					done()
				}
			}()
		}
		g.Stop()
		wg.Wait()
		if err := g.Wait(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
}

func TestLifetimeContextCancelsWithoutPreventingDrain(t *testing.T) {
	var g Group
	ctx := g.Context()
	g.Stop()
	if ctx.Err() != context.Canceled {
		t.Fatal("root not canceled")
	}
	if err := g.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	var stopped Group
	stopped.Stop()
	if stopped.Context().Err() != context.Canceled {
		t.Fatal("late root escaped cancellation")
	}
}
