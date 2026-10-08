//go:build linux && (arm64 || amd64)

package plugins

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

// The helper runs before the Talk server initializes. Only the thread which
// installs the restrictions executes the plugin; exec discards other threads.
func init() {
	if len(os.Args) != 3 || os.Args[1] != "--talk-plugin-sandbox-v1" {
		return
	}
	runtime.LockOSThread()
	if err := sandboxExec(os.Args[2]); err != nil {
		fmt.Fprintln(os.Stderr, "plugin sandbox:", err)
		os.Exit(126)
	}
	os.Exit(126)
}
func sandboxExec(executable string) error {
	executable, err := filepath.Abs(executable)
	if err != nil {
		return err
	}
	if err = unix.Prctl(unix.PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0); err != nil {
		return err
	}
	abi, _, errno := unix.Syscall(unix.SYS_LANDLOCK_CREATE_RULESET, 0, 0, 1)
	if errno != 0 {
		return fmt.Errorf("Landlock unavailable: %w", errno)
	}
	if abi < 3 {
		return fmt.Errorf("Landlock ABI 3 or newer required")
	}
	// Handle every supported filesystem permission, including truncate/refer.
	access := uint64((1 << 15) - 1)
	if abi >= 5 {
		access |= unix.LANDLOCK_ACCESS_FS_IOCTL_DEV
	}
	attr := struct{ Access uint64 }{access}
	fd, _, errno := unix.Syscall(unix.SYS_LANDLOCK_CREATE_RULESET, uintptr(unsafe.Pointer(&attr)), 8, 0)
	if errno != 0 {
		return errno
	}
	defer unix.Close(int(fd))
	pathFD, err := unix.Open(filepath.Dir(executable), unix.O_PATH|unix.O_CLOEXEC, 0)
	if err != nil {
		return err
	}
	defer unix.Close(pathFD)
	rule := struct {
		Access uint64
		Parent int32
		Pad    uint32
	}{unix.LANDLOCK_ACCESS_FS_READ_FILE | unix.LANDLOCK_ACCESS_FS_READ_DIR | unix.LANDLOCK_ACCESS_FS_EXECUTE, int32(pathFD), 0}
	_, _, errno = unix.Syscall6(unix.SYS_LANDLOCK_ADD_RULE, fd, 1, uintptr(unsafe.Pointer(&rule)), 0, 0, 0)
	if errno != 0 {
		return errno
	}
	_, _, errno = unix.Syscall(unix.SYS_LANDLOCK_RESTRICT_SELF, fd, 0, 0)
	if errno != 0 {
		return errno
	}
	if err = unix.Chdir(filepath.Dir(executable)); err != nil {
		return err
	}
	// Core dumps and open descriptors are bounded. All application data goes
	// through the broker, not a writable host mount.
	if err = unix.Setrlimit(unix.RLIMIT_CORE, &unix.Rlimit{}); err != nil {
		return err
	}
	if err = unix.Setrlimit(unix.RLIMIT_AS, &unix.Rlimit{Cur: 2 << 30, Max: 2 << 30}); err != nil {
		return err
	}
	if err = unix.Setrlimit(unix.RLIMIT_NOFILE, &unix.Rlimit{Cur: 64, Max: 64}); err != nil {
		return err
	}
	if err = installFilter(); err != nil {
		return err
	}
	return syscall.Exec(executable, []string{executable}, []string{"LANG=C.UTF-8", "GOMAXPROCS=2", "GOMEMLIMIT=128MiB"})
}
func installFilter() error {
	arch := uint32(unix.AUDIT_ARCH_AARCH64)
	if runtime.GOARCH == "amd64" {
		arch = unix.AUDIT_ARCH_X86_64
	}
	stmt := func(code uint16, k uint32) unix.SockFilter { return unix.SockFilter{Code: code, K: k} }
	eq := func(k uint32, jt, jf uint8) unix.SockFilter {
		return unix.SockFilter{Code: unix.BPF_JMP | unix.BPF_JEQ | unix.BPF_K, K: k, Jt: jt, Jf: jf}
	}
	allow := stmt(unix.BPF_RET|unix.BPF_K, unix.SECCOMP_RET_ALLOW)
	deny := stmt(unix.BPF_RET|unix.BPF_K, unix.SECCOMP_RET_ERRNO|uint32(unix.EPERM))
	f := []unix.SockFilter{stmt(unix.BPF_LD|unix.BPF_W|unix.BPF_ABS, 4), eq(arch, 1, 0), stmt(unix.BPF_RET|unix.BPF_K, unix.SECCOMP_RET_KILL_PROCESS), stmt(unix.BPF_LD|unix.BPF_W|unix.BPF_ABS, 0)}
	// Go needs threads, but subprocess creation and cross-process signals are
	// forbidden. Test clone's CLONE_THREAD bit and tgkill's process ID.
	f = append(f, eq(uint32(unix.SYS_CLONE), 0, 4), stmt(unix.BPF_LD|unix.BPF_W|unix.BPF_ABS, 16), unix.SockFilter{Code: unix.BPF_JMP | unix.BPF_JSET | unix.BPF_K, K: unix.CLONE_THREAD, Jt: 1}, deny, allow)
	f = append(f, eq(uint32(unix.SYS_TGKILL), 0, 4), stmt(unix.BPF_LD|unix.BPF_W|unix.BPF_ABS, 16), eq(uint32(os.Getpid()), 1, 0), deny, allow)
	calls := []uint32{unix.SYS_READ, unix.SYS_WRITE, unix.SYS_CLOSE, unix.SYS_FSTAT, unix.SYS_LSEEK, unix.SYS_MMAP, unix.SYS_MPROTECT, unix.SYS_MUNMAP, unix.SYS_BRK, unix.SYS_RT_SIGACTION, unix.SYS_RT_SIGPROCMASK, unix.SYS_RT_SIGRETURN, unix.SYS_SIGALTSTACK, unix.SYS_FUTEX, unix.SYS_SCHED_YIELD, unix.SYS_GETPID, unix.SYS_GETTID, unix.SYS_CLOCK_GETTIME, unix.SYS_NANOSLEEP, unix.SYS_CLOCK_NANOSLEEP, unix.SYS_GETRANDOM, unix.SYS_MADVISE, unix.SYS_EXIT, unix.SYS_EXIT_GROUP, unix.SYS_EXECVE, unix.SYS_FCNTL, unix.SYS_DUP, unix.SYS_DUP3, unix.SYS_PIPE2, unix.SYS_OPENAT, unix.SYS_GETDENTS64, unix.SYS_READLINKAT, unix.SYS_STATX, unix.SYS_UNAME, unix.SYS_SCHED_GETAFFINITY, unix.SYS_EPOLL_CREATE1, unix.SYS_EPOLL_CTL, unix.SYS_EPOLL_PWAIT, unix.SYS_EVENTFD2, unix.SYS_RESTART_SYSCALL, unix.SYS_SET_TID_ADDRESS, unix.SYS_SET_ROBUST_LIST, unix.SYS_GETUID, unix.SYS_GETEUID, unix.SYS_GETGID, unix.SYS_GETEGID}
	calls = append(calls, archSyscalls()...)
	for _, n := range calls {
		f = append(f, eq(n, 0, 1), allow)
	}
	f = append(f, deny)
	prog := unix.SockFprog{Len: uint16(len(f)), Filter: &f[0]}
	return unix.Prctl(unix.PR_SET_SECCOMP, unix.SECCOMP_MODE_FILTER, uintptr(unsafe.Pointer(&prog)), 0, 0)
}
