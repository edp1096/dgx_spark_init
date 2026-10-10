"""Read host, CUDA-process and container memory while an isolated reference runs."""
import argparse,json,subprocess,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    started=time.monotonic()
    with a.out.open('w') as f:
        while True:
            r=subprocess.run(['docker','inspect',a.name],capture_output=True,text=True)
            if r.returncode:return
            info=json.loads(r.stdout)[0];pid=info['State']['Pid']
            mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if len(l.split())>=3}
            vm=dict(l.split() for l in Path('/proc/vmstat').read_text().splitlines())
            row={'seconds':time.monotonic()-started,'host':{k:mem[k] for k in ['MemAvailable','MemFree','SwapTotal','SwapFree']},'swap_io_pages':{k:int(vm[k]) for k in ['pswpin','pswpout']},'oom':info['State']['OOMKilled']}
            if pid:
                try:
                    cg=Path('/sys/fs/cgroup')/Path(f'/proc/{pid}/cgroup').read_text().split('::',1)[1].strip().lstrip('/')
                    row['cgroup']={k:(cg/k).read_text().strip() for k in ['memory.current','memory.peak','memory.swap.current','memory.events']}
                    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                    row['gpu_processes']=gpu.stdout.strip()
                except (OSError,subprocess.TimeoutExpired):pass
            f.write(json.dumps(row)+'\n');f.flush()
            if not pid:return
            time.sleep(2)
if __name__=='__main__':main()
