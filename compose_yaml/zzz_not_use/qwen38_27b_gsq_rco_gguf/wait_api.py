#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('url')
parser.add_argument('--timeout', type=float, default=900)
args = parser.parse_args()
deadline = time.monotonic() + args.timeout
print('Waiting for API:', args.url, flush=True)
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(args.url, timeout=5) as response:
            value = json.load(response)
            if response.status == 200:
                print('API ready:', args.url, flush=True)
                break
    except (OSError, ValueError):
        pass
    time.sleep(2)
else:
    raise SystemExit('API readiness timeout: ' + args.url)
