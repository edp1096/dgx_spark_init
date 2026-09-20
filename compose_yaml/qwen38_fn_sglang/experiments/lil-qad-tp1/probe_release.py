"""Default two-request profile plus diverse near-limit retrieval."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    subprocess.run([sys.executable,str(HERE/'smoke_stage2.py'),
        '--output',str(args.output)],check=True)
    subprocess.run([sys.executable,str(HERE/'probe_context.py'),
        '--output',str(args.output.with_name('diverse-64k.json')),
        '--targets','65536','--diverse'],check=True)
