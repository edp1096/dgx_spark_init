#!/usr/bin/env python3
"""Compare existing local weights only. No hub clients, downloads or GPU use."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weights_core.planning import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, help='fp8, ornith_15, gemma4_26b, or a local profile JSON path')
    for name in ('original', 'donor', 'base', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    # Never overwrite source metadata or a prior report.
    if args.report.exists():
        parser.error('Report already exists')
    report = audit(args.original, args.donor, args.base, args.profile)
    with args.report.open('x') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(f"Compared: {report['unchanged_source_tensors']} unchanged, {len(report['changes'])} changed")
    print(f'Plan: {args.report} (not a converted model)')


if __name__ == '__main__':
    main()
