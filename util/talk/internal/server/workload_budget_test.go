package server

import "testing"

func TestNemotronRequestBudgetUsesDecodedLengthAndDiarization(t *testing.T) {
	for _, tc := range []struct {
		seconds     int64
		plain, diar float64
	}{{120, 4, 4.25}, {600, 4.5, 4.75}, {3600, 7.25, 7.5}} {
		bytes := tc.seconds*16000*2 + 78
		if got := nemotronRequestBudget(bytes, false); got != tc.plain {
			t.Fatalf("%ds plain: %v", tc.seconds, got)
		}
		if got := nemotronRequestBudget(bytes, true); got != tc.diar {
			t.Fatalf("%ds diar: %v", tc.seconds, got)
		}
	}
	if nemotronRequestBudget(3600*32000, true) <= 6.97 {
		t.Fatal("budget below observed hour-long peak")
	}
	if nemotronRequestBudget(4*3600*32000, true) <= nemotronRequestBudget(3600*32000, true) {
		t.Fatal("long inputs silently capped")
	}
}
