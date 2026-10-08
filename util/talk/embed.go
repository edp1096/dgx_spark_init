package chat

import "embed"

// WebDist contains the production Svelte build. The Makefile creates it before
// compiling the Go binary.
//
//go:embed all:web/dist
var WebDist embed.FS
