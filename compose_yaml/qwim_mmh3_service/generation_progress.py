"""Per-request progress and estimates learned from completed fixed profiles."""
import json
import math
import os
from pathlib import Path
import statistics
import time

STAGES = ['encode', 'release_encoder', 'sample', 'release_sample_workspace',
          'decode_video', 'release_video_vae', 'decode_audio', 'release_audio_vae', 'save']
PROFILE = {'h3': 'h3-864x480-124-24-20-v1', 'qwim': 'qwim-1024x1024-40-v1'}


def stages(kind):
    return [s for s in STAGES if kind == 'h3' or s not in ('decode_audio', 'release_audio_vae')]


def append_event(root, request_id, event):
    folder = Path(root) / 'progress'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / (request_id + '.jsonl')).open('a') as file:
        file.write(json.dumps(event) + '\n')


def read_events(root, request_id, after):
    path = Path(root) / 'progress' / (request_id + '.jsonl')
    if not path.exists():
        return {'events': [], 'next': 0}
    # A concurrent append may leave an incomplete last line. Do not consume it.
    lines = path.read_text().splitlines(keepends=True)
    if lines and not lines[-1].endswith('\n'):
        lines.pop()
    end = min(len(lines), after + 128)
    return {'events': [json.loads(line) for line in lines[after:end]], 'next': end}


def prune(root):
    folder = Path(root) / 'progress'
    if folder.exists():
        for path in folder.glob('*.jsonl'):
            if time.time() - path.stat().st_mtime > 3600:
                path.unlink(missing_ok=True)


class Tracker:
    def __init__(self, root, state, request_id, kind, clock=time.monotonic):
        self.root, self.state, self.request_id, self.kind = Path(root), Path(state), request_id, kind
        self.clock = clock
        self.started = clock()
        self.phase_started = self.started
        self.phase = 'encode'
        self.step = self.total = None
        self.samples = []
        self.completed = set()
        self.expected = {}
        try:
            history = json.loads((self.state / 'timings.json').read_text()).get(PROFILE[kind], [])
            for phase in stages(kind):
                values = [r[phase] for r in history if isinstance(r.get(phase), (int, float)) and math.isfinite(r[phase]) and r[phase] >= 0]
                if values:
                    self.expected[phase] = statistics.median(values)
        except (OSError, ValueError, TypeError, KeyError):
            pass

    def record(self, event, **data):
        now = self.clock()
        if event.endswith('_start') and event[:-6] in stages(self.kind):
            self.phase = event[:-6]
            self.phase_started = now
            self.step = self.total = None
            self.samples = []
        if event.endswith('_end'):
            self.completed.add(event[:-4])
        if event == 'progress' and data.get('total', 0) > 0:
            self.step, self.total = int(data['step']), int(data['total'])
            if self.step > 0 and (not self.samples or self.step > self.samples[-1][0]):
                self.samples.append((self.step, now - self.phase_started))
                self.samples = self.samples[-6:]
        remaining, scope = self.estimate(now)
        if event == 'idle':
            remaining, scope = 0.0, 'total'
        item = {'event': event, 'stage': self.phase, 'kind': self.kind, 'time': time.time(),
                'elapsed_seconds': round(now - self.started, 2), 'eta_seconds': remaining,
                'eta_scope': scope, 'step': self.step, 'total': self.total}
        try:
            append_event(self.root, self.request_id, item)
        except OSError:
            pass
        return item

    def estimate(self, now):
        order = stages(self.kind)
        future = order[order.index(self.phase) + 1:]
        current = None
        if self.phase in self.completed:
            current = 0.0
        elif self.samples and self.step is not None:
            first, last = self.samples[0], self.samples[-1]
            speed = ((last[1] - first[1]) / (last[0] - first[0])
                     if last[0] > first[0] else last[1] / last[0])
            current = max(0.0, speed * (self.total - self.step))
        elif self.phase in self.expected:
            current = max(0.0, self.expected[self.phase] - (now - self.phase_started))
        if current is None:
            return None, None
        full = all(s in self.expected for s in future)
        remaining = current + sum(self.expected[s] for s in future) if full else current
        # An overrun is not a completed job. Avoid an indefinite "0 seconds" ETA.
        if remaining < 1 and self.phase not in self.completed and event_incomplete(self.step, self.total):
            return None, None
        return round(remaining, 1), 'total' if full else 'stage'

    def remember(self, timings):
        self.state.mkdir(parents=True, exist_ok=True)
        path = self.state / 'timings.json'
        try:
            history = json.loads(path.read_text())
        except (OSError, ValueError):
            history = {}
        rows = history.setdefault(PROFILE[self.kind], [])
        rows.append({s: float(timings[s]) for s in stages(self.kind)})
        history[PROFILE[self.kind]] = rows[-8:]
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(history))
        os.replace(temp, path)


def event_incomplete(step, total):
    return step is None or total is None or step < total
