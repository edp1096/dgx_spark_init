//go:build linux

package plugins

import "golang.org/x/sys/unix"

func archSyscalls() []uint32 { return []uint32{unix.SYS_FSTATAT} }
