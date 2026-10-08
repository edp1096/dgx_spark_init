//go:build linux

package plugins

import "golang.org/x/sys/unix"

func archSyscalls() []uint32 {
	return []uint32{unix.SYS_NEWFSTATAT, unix.SYS_ARCH_PRCTL, unix.SYS_READLINK, unix.SYS_EPOLL_WAIT}
}
