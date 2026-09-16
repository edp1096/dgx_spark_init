"""Bounded native buffered Engram reads, with persistent workers and no math changes.

Does not bypass the OS page cache or change row order. The
caller owns fds and buffers until read() completes; all workers join on errors.
"""
import atexit
import ctypes as C
import hashlib
import os
from pathlib import Path
import subprocess
import threading

_SOURCE = r'''
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <stdatomic.h>
#include <limits.h>

typedef struct { int fd; int64_t base, width, count; const int64_t *rows; unsigned char *out; } Job;
typedef struct { const Job *job; int64_t first, last; } Task;
typedef struct {
    pthread_mutex_t mu; pthread_cond_t work, done;
    pthread_t *threads; int n, stop, active; uint64_t epoch;
    Task *tasks; size_t capacity, count; atomic_size_t next; atomic_int error;
} Reader;
static void execute(Reader *r) {
    for (;;) {
        size_t t = atomic_fetch_add(&r->next, 1);
        if (t >= r->count || atomic_load(&r->error)) return;
        Task *task = &r->tasks[t]; const Job *j = task->job;
        for (int64_t row = task->first; row < task->last; row++) {
            size_t got = 0;
            while (got < (size_t)j->width) {
                ssize_t n = pread(j->fd, j->out + row*j->width + got,
                    (size_t)j->width-got, j->base + j->rows[row]*j->width + got);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) { int expected = 0;
                    atomic_compare_exchange_strong(&r->error, &expected, n < 0 ? errno : EIO);
                    return;
                }
                got += (size_t)n;
            }
        }
    }
}
static void *worker(void *arg) {
    Reader *r = arg; uint64_t seen = 0;
    pthread_mutex_lock(&r->mu);
    for (;;) {
        while (!r->stop && seen == r->epoch) pthread_cond_wait(&r->work, &r->mu);
        if (r->stop) break;
        seen = r->epoch;
        pthread_mutex_unlock(&r->mu); execute(r); pthread_mutex_lock(&r->mu);
        if (--r->active == 0) pthread_cond_signal(&r->done);
    }
    pthread_mutex_unlock(&r->mu); return NULL;
}
void er_free(Reader *r) {
    if (!r) return;
    pthread_mutex_lock(&r->mu); r->stop = 1; pthread_cond_broadcast(&r->work); pthread_mutex_unlock(&r->mu);
    for (int i=0; i<r->n; i++) pthread_join(r->threads[i], NULL);
    pthread_cond_destroy(&r->work); pthread_cond_destroy(&r->done); pthread_mutex_destroy(&r->mu);
    free(r->threads); free(r->tasks); free(r);
}
Reader *er_create(int threads) {
    if (threads < 1 || threads > 64) return NULL;
    Reader *r = calloc(1, sizeof(*r)); if (!r) return NULL;
    atomic_init(&r->next, 0); atomic_init(&r->error, 0);
    pthread_mutex_init(&r->mu, NULL); pthread_cond_init(&r->work, NULL); pthread_cond_init(&r->done, NULL);
    r->threads = calloc(threads, sizeof(pthread_t)); if (!r->threads) { er_free(r); return NULL; }
    for (int i=0; i<threads; i++) {
        if (pthread_create(&r->threads[i], NULL, worker, r)) { er_free(r); return NULL; }
        r->n++;
    }
    return r;
}
int er_read(Reader *r, const Job *jobs, size_t n_jobs, int64_t chunk) {
    if (!r || chunk < 1) return EINVAL;
    size_t count = 0;
    for (size_t j=0; j<n_jobs; j++) {
        if (jobs[j].count < 0 || jobs[j].width < 1 || jobs[j].base < 0) return EINVAL;
        if ((uint64_t)jobs[j].count > SIZE_MAX / (uint64_t)jobs[j].width) return EOVERFLOW;
        size_t tasks = jobs[j].count/chunk + (jobs[j].count%chunk != 0);
        if (tasks > SIZE_MAX-count) return EOVERFLOW;
        count += tasks;
        for (int64_t i=0; i<jobs[j].count; i++)
            if (jobs[j].rows[i] < 0 || jobs[j].rows[i] > (INT64_MAX-jobs[j].base-jobs[j].width)/jobs[j].width)
                return EOVERFLOW;
    }
    if (!count) return 0;
    if (count > SIZE_MAX/sizeof(Task)) return EOVERFLOW;
    if (count > r->capacity) {
        Task *tasks = realloc(r->tasks, count*sizeof(Task)); if (!tasks) return ENOMEM;
        r->tasks = tasks; r->capacity = count;
    }
    size_t t=0;
    for (size_t j=0; j<n_jobs; j++)
        for (int64_t i=0; i<jobs[j].count;) {
            int64_t end = jobs[j].count-i < chunk ? jobs[j].count : i+chunk;
            r->tasks[t++] = (Task){&jobs[j], i, end}; i=end;
        }
    r->count=count; atomic_store(&r->next,0); atomic_store(&r->error,0);
    if (count == 1) { execute(r); return atomic_load(&r->error); }
    pthread_mutex_lock(&r->mu); r->active=r->n; r->epoch++;
    pthread_cond_broadcast(&r->work);
    while (r->active) pthread_cond_wait(&r->done,&r->mu);
    pthread_mutex_unlock(&r->mu); return atomic_load(&r->error);
}
'''

class _Job(C.Structure):
    _fields_ = [('fd', C.c_int), ('base', C.c_int64), ('width', C.c_int64),
                ('count', C.c_int64), ('rows', C.POINTER(C.c_int64)), ('out', C.c_void_p)]


_LOAD_LOCK = threading.Lock()


def _load_unlocked(source=_SOURCE):
    root = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache')))/'ds41-engram-reader'
    root.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256((source + os.uname().machine).encode()).hexdigest()[:20]
    library = root/f'{tag}.so'
    if not library.exists():
        tmp = root/f'{tag}.{os.getpid()}.so'
        try:
            subprocess.run(['cc','-O3','-std=c11','-shared','-fPIC','-pthread','-x','c','-','-o',str(tmp)],
                           input=source,text=True,check=True,capture_output=True)
            os.replace(tmp,library)
        finally:
            tmp.unlink(missing_ok=True)
    lib = C.CDLL(str(library))
    lib.er_create.argtypes=[C.c_int]; lib.er_create.restype=C.c_void_p
    lib.er_free.argtypes=[C.c_void_p]; lib.er_free.restype=None
    lib.er_read.argtypes=[C.c_void_p,C.POINTER(_Job),C.c_size_t,C.c_int64]; lib.er_read.restype=C.c_int
    return lib


def _load(source=_SOURCE):
    with _LOAD_LOCK:
        return _load_unlocked(source)


class Reader:
    def __init__(self, threads=32, *, _source=_SOURCE):
        if not 1 <= threads <= 64: raise ValueError('reader threads must be 1..64')
        self._lock=threading.Lock(); self._lib=_load(_source); self._pid=os.getpid()
        self._ptr=self._lib.er_create(threads)
        if not self._ptr: raise RuntimeError('native reader worker creation failed')
        atexit.register(self.close)

    def close(self):
        if os.getpid() != self._pid: return
        with self._lock:
            if self._ptr:
                self._lib.er_free(self._ptr); self._ptr=None

    def read(self, jobs, chunk=16):
        if os.getpid() != self._pid: raise RuntimeError('reader cannot be reused after fork')
        if not 1 <= chunk <= 2**31: raise ValueError('invalid row chunk')
        args=(_Job*len(jobs))(); keep=[]
        for i,(fd,base,rows,width,buf) in enumerate(jobs):
            if base < 0 or width < 1 or base+width > 2**63-1: raise ValueError('invalid row geometry')
            if any(row < 0 or row > (2**63-1-base-width)//width for row in rows):
                raise ValueError('row offset out of range')
            view=memoryview(buf).cast('B')
            if view.readonly or len(view) != len(rows)*width: raise ValueError('invalid output buffer')
            ids=(C.c_int64*len(rows))(*rows)
            output=(C.c_ubyte*len(view)).from_buffer(view)
            keep.extend((view,ids,output))
            args[i]=_Job(fd,base,width,len(rows),ids,C.addressof(output))
        with self._lock:
            if not self._ptr: raise RuntimeError('reader is closed')
            error=self._lib.er_read(self._ptr,args,len(jobs),chunk)
        if error: raise OSError(error,os.strerror(error))

def advise_small_jobs(jobs):
    if sum(len(j[2]) for j in jobs)>512:return
    page=os.sysconf('SC_PAGE_SIZE')
    for fd,base,rows,width,buf in jobs:
        if base<0 or width<1 or base+width>2**63-1:continue
        if any(row<0 or row>(2**63-1-base-width)//width for row in rows):continue
        pages={offset for row in rows for offset in
               range((base+row*width)//page,(base+(row+1)*width-1)//page+1)}
        for offset in sorted(pages):
            if 0<=offset*page<2**63-page:
                try:
                    os.posix_fadvise(fd,offset*page,page,os.POSIX_FADV_WILLNEED)
                except OSError:
                    pass  # Optional hint; authoritative pread still reports errors.


class AdvisedReader(Reader):
    """Same pread pool, with bounded current-row I/O hints for small batches."""
    def read(self,jobs,chunk=16):
        advise_small_jobs(jobs)
        return super().read(jobs,chunk)
