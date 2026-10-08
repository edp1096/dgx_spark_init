"""Offline demand-I/O replay; not a GPU speed benchmark or runtime change."""
import argparse, json, hashlib, math
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def run(rows, profile, scratch=0):
    capacity = 224 - scratch
    caches = {l: OrderedDict((e, None) for e in reversed(profile.get(l, [])[:capacity])) for l in range(40)}
    stats = {phase: Counter() for phase in ('pp', 'tg')}
    preload = sum(map(len, caches.values()))

    def ensure(cache, needed, stat):
        requested = set(needed)
        missing = []
        for e in dict.fromkeys(needed):
            if e in cache:
                cache.move_to_end(e)
                stat['hits'] += 1
            else:
                missing.append(e)
        free = capacity - len(cache)
        for _ in missing:
            if free:
                free -= 1
            else:
                victim = next(e for e in cache if e not in requested)
                del cache[victim]
        for e in missing:
            cache[e] = None
        stat['misses'] += len(missing)
        stat['groups'] += 1
        assert len(cache) <= capacity

    for row in rows:
        layer, tokens = row['layer'], row['tokens']
        if layer >= 40:
            continue  # Draft experts are fully resident in all candidates.
        # Trace lacks scheduler phase metadata. <=16 includes decode/spec verify
        # and may include a short prefill tail; this is a diagnostic proxy only.
        phase = 'pp' if tokens > 16 else 'tg'
        stat = stats[phase]
        cache = caches[layer]
        needed = row['needed']
        unique = set(needed)
        stat['calls'] += 1
        if scratch and phase == 'pp':
            # Proposed fixed-budget split: transient PP misses do not enter hot
            # cache. Groups execute sequentially, with scratch reuse fenced.
            hits = unique.intersection(cache)
            stat['hits'] += len(hits)
            stat['misses'] += len(unique - hits)
            stat['groups'] += bool(hits) + math.ceil(len(unique - hits) / scratch)
        elif len(unique) <= capacity:
            ensure(cache, needed, stat)
        else:
            topk = len(needed) // tokens
            freq = Counter(needed[-64 * topk:])
            resident = unique.intersection(cache)
            cold = sorted(unique - resident, key=lambda e: (freq[e], e))
            groups = [sorted(resident, key=lambda e: (freq[e], e))] if resident else []
            groups += [cold[i:i+capacity] for i in range(0, len(cold), capacity)]
            for group in groups:
                ensure(cache, group, stat)
    result = {p: dict(s) for p, s in stats.items()}
    result['preload_experts'] = preload
    result['preload_gib_per_rank'] = preload * 9400320 / 2**30
    result['total_reads_including_preload'] = preload + sum(s['misses'] for s in stats.values())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True); p.add_argument('--test', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    train, test = read(args.train), read(args.test)
    counts = defaultdict(Counter)
    for row in train:
        if row['layer'] < 40 and row['tokens'] <= 16:
            counts[row['layer']].update(row['needed'])
    profile = {l: [e for e, _ in c.most_common()] for l, c in counts.items()}
    variants = {'current_cold': run(test, {}), 'seeded_224': run(test, profile)}
    for count in (64, 128):
        variants[f'seeded_{count}'] = run(test, {l: ids[:count] for l, ids in profile.items()})
    for scratch in (32, 64):
        variants[f'seeded_split_{224-scratch}_{scratch}'] = run(test, profile, scratch)
    result = {'kind': 'offline route replay, NOT speed measurements',
              'source': {k: {'file': Path(v).name, 'sha256': hashlib.sha256(Path(v).read_bytes()).hexdigest()} for k, v in [('train', args.train), ('test', args.test)]},
              'limitations': ['Historical traces, not the current 4096-scheduler workload.', 'Phase inferred from tokens >16; short prefill tails may be classified as TG.', 'Demand misses/group counts omit kernel time and async overlap.', 'Static split does not adapt/promote during PP; one candidate policy, not an optimum.'],
              'variants': variants}
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(variants, indent=2))

if __name__ == '__main__': main()
