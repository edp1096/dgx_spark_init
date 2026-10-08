"""Run upstream unchanged, then publish readable report links on exit."""
import signal
import subprocess
import sys

from reports import ROOT, catalog


def main():
    process = subprocess.Popen(['tool-eval-bench', *sys.argv[1:]])

    def forward(signum, _frame):
        if process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    code = process.wait()
    try:
        rows = catalog(ROOT)
        if rows:
            print(f'Report links updated: runs/named/ ({len(rows)} reports)', flush=True)
    except Exception as exc:
        print(f'Report links failed: {exc}', file=sys.stderr, flush=True)
        if code == 0:
            code = 1
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    sys.exit(main())
