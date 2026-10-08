package attachment

import "testing"

func TestCommonAndOptionalOverrides(t *testing.T) {
	l := (Limits{}).Normalized()
	for _, k := range []string{"image", "audio", "video", "document"} {
		if l.ForType(k) != 256*MiB {
			t.Fatalf("%s did not inherit default", k)
		}
	}
	if len(l.TypeLimitsMB) != 0 {
		t.Fatal("defaults created per-type overrides")
	}
	l.MaxFileMB = 128
	l.TypeLimitsMB = map[string]int{"image": 15, "video": 512, "document": 0}
	l = l.Normalized()
	if l.ForType("document") != 128*MiB || l.ForType("audio") != 128*MiB || l.ForType("image") != 15*MiB || l.ForType("video") != 512*MiB {
		t.Fatal("wrong override inheritance")
	}
	l.MaxFileMB = 256
	if l.ForType("audio") != 256*MiB || l.ForType("image") != 15*MiB {
		t.Fatal("common change altered explicit override")
	}
	if l.MessageBytes() < l.MaxBytes() {
		t.Fatal("message size prevents largest file")
	}
	delete(l.TypeLimitsMB, "image")
	if l.ForType("image") != 256*MiB {
		t.Fatal("cleared override did not inherit")
	}
	for _, bad := range []Limits{{MaxFileMB: -1}, {MaxFileMB: 4097}, {MaxFiles: 21}, {TypeLimitsMB: map[string]int{"unknown": 12}}, {TypeLimitsMB: map[string]int{"image": -1}}} {
		if bad.Validate() == nil {
			t.Fatalf("invalid policy accepted: %+v", bad)
		}
	}
}
