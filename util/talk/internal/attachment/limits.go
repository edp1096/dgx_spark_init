// Package attachment defines the shared upload and generated-file policy.
package attachment

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
)

//go:embed defaults.json
var defaultsJSON []byte
var defaults = func() Limits {
	var l Limits
	if err := json.Unmarshal(defaultsJSON, &l); err != nil {
		panic(err)
	}
	return l
}()

func Default() Limits { return defaults }

const MiB int64 = 1 << 20

// TypeLimitsMB is optional: an absent/zero entry inherits MaxFileMB.
// Explicit type limits may be smaller or larger than the common default.
type Limits struct {
	MaxFileMB    int            `yaml:"max_file_mb" json:"max_file_mb"`
	MaxFiles     int            `yaml:"max_files" json:"max_files"`
	TypeLimitsMB map[string]int `yaml:"type_limits_mb,omitempty" json:"type_limits_mb,omitempty"`
}

func (l Limits) Normalized() Limits {
	if l.MaxFileMB == 0 {
		l.MaxFileMB = defaults.MaxFileMB
	}
	if l.MaxFiles == 0 {
		l.MaxFiles = defaults.MaxFiles
	}
	overrides := make(map[string]int, len(l.TypeLimitsMB))
	for k, v := range l.TypeLimitsMB {
		if v != 0 {
			overrides[k] = v
		}
	}
	l.TypeLimitsMB = overrides
	return l
}
func (l Limits) Validate() error {
	l = l.Normalized()
	if l.MaxFileMB < 1 || l.MaxFileMB > 4096 || l.MaxFiles < 1 || l.MaxFiles > 20 {
		return fmt.Errorf("attachments: max_file_mb must be 1–4096 and max_files 1–20")
	}
	for k, v := range l.TypeLimitsMB {
		if (k != "image" && k != "audio" && k != "video" && k != "document") || v < 1 || v > 4096 {
			return fmt.Errorf("attachments.type_limits_mb: invalid type or size for %q", k)
		}
	}
	return nil
}
func (l Limits) ForType(kind string) int64 {
	l = l.Normalized()
	v := l.MaxFileMB
	if n := l.TypeLimitsMB[kind]; n > 0 {
		v = n
	}
	return int64(v) * MiB
}
func Kind(mime string) string {
	for _, kind := range []string{"image", "audio", "video"} {
		if strings.HasPrefix(mime, kind+"/") {
			return kind
		}
	}
	return "document"
}
func (l Limits) ForMIME(mime string) int64 { return l.ForType(Kind(mime)) }
func (l Limits) MaxBytes() int64 {
	l = l.Normalized()
	v := l.MaxFileMB
	for _, n := range l.TypeLimitsMB {
		if n > v {
			v = n
		}
	}
	return int64(v) * MiB
}
func (l Limits) MessageBytes() int64 { return l.MaxBytes() * int64(l.Normalized().MaxFiles) }
