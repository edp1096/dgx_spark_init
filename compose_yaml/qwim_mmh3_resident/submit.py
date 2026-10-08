"""Submit a serial generation request to an already running resident worker."""
import argparse
import json
from pathlib import Path
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_directory', type=Path)
    parser.add_argument('kind', choices=('h3', 'qwim'))
    parser.add_argument('prompt')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--timeout', type=float, default=1200)
    args = parser.parse_args()
    if not args.prompt.strip() or not 0 <= args.seed <= 2**64 - 1:
        parser.error('provide a nonempty prompt and a uint64 seed')
    if not (args.job_directory / 'ready.json').exists():
        parser.error('worker has not reported ready in this job directory')
    case = args.kind + '-' + uuid.uuid4().hex
    request = dict(case=case, kind=args.kind, prompt=args.prompt, seed=args.seed)
    path = args.job_directory / 'requests' / (case + '.tmp')
    path.write_text(json.dumps(request))
    path.rename(path.with_suffix('.json'))
    result = args.job_directory / 'results' / (case + '.json')
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if result.exists():
            data = json.loads(result.read_text())
            print(json.dumps(data, indent=2))
            return 0 if data['status'] == 'success' else 1
        time.sleep(.2)
    print(f'Timed out waiting for {case}; the queued request was not canceled.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
