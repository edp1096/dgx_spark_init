"""Full API regressions, diverse PLE workload, then exact 1M-context probe."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for script, output, extra in (
        ('smoke_stage2.py', args.output.with_name('smoke-results.json'), []),
        ('probe_context.py', args.output.with_name('diverse-results.json'),
         ['--targets','131072','--diverse']),
        ('probe_context.py', args.output, ['--targets','1048576']),
    ):
        subprocess.run([sys.executable,str(HERE/script),'--output',str(output),*extra],check=True)
    from benchmark_serving import run
    run(args.output.with_name('serving-benchmark.json'))


if __name__ == '__main__':
    main()
