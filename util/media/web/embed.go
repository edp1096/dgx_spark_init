package webassets

import (
	"embed"
	"io/fs"
)

//go:embed dist/*
var content embed.FS

func Files() fs.FS { sub, _ := fs.Sub(content, "dist"); return sub }
