#!/usr/bin/env python3
"""Readable report links and an interactive HTML comparison picker."""
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def slug(value):
    return re.sub(r'[^a-z0-9_-]+', '-', value.lower()).strip('-_')[:90] or 'model'


def field(text, name):
    match = re.search(r'^\| ' + re.escape(name) + r' \| (.*?) \|$', text, re.M)
    return match[1].strip('` ') if match else ''


def catalog(root):
    runs = root / 'runs'
    named = runs / 'named'
    named.mkdir(parents=True, exist_ok=True)
    sources = sorted(runs.glob('[0-9][0-9][0-9][0-9]/[0-9][0-9]/*.md'))
    documents = {p: p.read_text() for p in sources}
    trials = {}
    for text in documents.values():
        for number, path in re.findall(r'^- Trial (\d+): `([^`]+)`', text, re.M):
            trials[Path(path).name] = int(number)
    result = []
    for source, text in documents.items():
        run = re.search(r'\*\*Run ID\*\*: `([^`]+)`', text)
        if not run:
            continue
        run_id = run[1]
        try:
            started = datetime.strptime(run_id.split('_')[0], '%Y-%m-%dT%H-%M-%S.%fZ')
        except ValueError:
            continue
        # Keep UTC explicit and microseconds to avoid colliding runs in one minute.
        stamp = started.strftime('%Y%m%d-%H%M%S-%fZ')
        model = field(text, 'Model (API)') or text.splitlines()[0].split(' — ')[-1]
        model_root = field(text, 'Model (Root)')
        name = slug(model)
        if 'ablit' in model_root.lower() and 'ablit' not in name:
            name += '-ablit'
        summary = text.startswith('# Cross-Trial Summary')
        kind = 'summary' if summary else f'trial{trials[source.name]}' if source.name in trials else 'run'
        score = re.search(r'\*\*Final Score\*\*: \*\*([\d.]+)\*\*', text)
        if summary:
            score = re.search(r'\| \*\*Final Score\*\* .*?\*\*([\d.]+) ±', text)
        link = named / f'{name}_{stamp}_{kind}.md'
        target = os.path.relpath(source, named)
        if link.is_symlink():
            if os.readlink(link) != target:
                raise RuntimeError(f'Conflicting report link: {link}')
        elif link.exists():
            raise RuntimeError(f'Refusing to overwrite file: {link}')
        else:
            link.symlink_to(target)
        # Once the summary identifies the trial number, retire only our own run alias.
        old = named / f'{name}_{stamp}_run.md'
        if kind.startswith('trial') and old.is_symlink() and os.readlink(old) == target:
            old.unlink()
        result.append({'path': link, 'model': name, 'time': started.strftime('%Y-%m-%d %H:%M:%S UTC'),
                       'score': score[1] if score else '?', 'kind': kind,
                       'suite': field(text, 'Scenarios') or '?', 'id': run_id})
    return list(reversed(result))


def compare_files(paths):
    if len(paths) != 2:
        raise ValueError('Specify exactly two Markdown report paths.')
    selected = []
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file() or path.suffix != '.md':
            raise ValueError(f'Markdown report not found: {value}')
        if not path.is_relative_to((ROOT / 'runs').resolve()):
            raise ValueError(f'Report must be inside runs/: {value}')
        text = path.read_text()
        model = field(text, 'Model (API)') or text.splitlines()[0].split(' — ')[-1]
        name = slug(model)
        if 'ablit' in field(text, 'Model (Root)').lower() and 'ablit' not in name:
            name += '-ablit'
        selected.append({'path': path, 'model': name})
    left, right = selected
    if left['path'] == right['path']:
        raise ValueError('Select two different reports.')
    # Resolve named links so upstream can identify the original summary suffix.
    args = [str(row['path'].relative_to(ROOT)) for row in selected]
    output = ROOT / 'runs' / (f"compare_{left['model'][:60]}_vs_{right['model'][:60]}_"
                             + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.html')
    subprocess.run([str(ROOT / 'manage.sh'), 'cli', 'compare', '--report', *args,
                    '-o', str(output.relative_to(ROOT))], cwd=ROOT, check=True)
    print(f'HTML: {output}')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'compare-files':
        compare_files(sys.argv[2:])
        return
    rows = catalog(ROOT)
    for i, row in enumerate(rows, 1):
        print(f"{i:2}. {row['model']} | {row['time']} | {row['kind']} | {row['score']}/100 | {row['suite']}")
    if not rows:
        print('No completed reports yet.')
        return
    if len(sys.argv) < 2 or sys.argv[1] != 'compare':
        print('\nFiles: runs/named/ (UTC; trial numbers appear after the summary is saved)')
        return
    if not sys.stdin.isatty():
        raise ValueError('Interactive terminal required; or use ./manage.sh compare RUN_ID_A RUN_ID_B')
    picks = input('Select two report numbers (e.g. 1 3): ').split()
    if len(picks) != 2 or not all(p.isdigit() and 1 <= int(p) <= len(rows) for p in picks):
        raise ValueError('Select two numbers from the list.')
    left, right = (rows[int(p) - 1] for p in picks)
    compare_files([left['path'], right['path']])


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
