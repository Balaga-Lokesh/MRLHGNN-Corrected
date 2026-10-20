"""Merge results.json files from several train.py runs (e.g. seeds split across terminals).

Also reports a seed ensemble: the held-out probabilities of all seeds averaged per drug-disease pair.
Every seed's prediction for a pair comes from a model whose training fold excluded that pair, so the
ensemble is still a held-out evaluation.

Usage: python aggregate.py runs/final   ->  writes runs/final/RESULTS.md and runs/final/summary.json
"""
import glob
import json
import os
import sys

import numpy as np

from metrics import paper_metrics, sk_metrics

root = sys.argv[1] if len(sys.argv) > 1 else 'runs/final'
runs, configs = [], []
for f in sorted(glob.glob(os.path.join(root, '**', 'results.json'), recursive=True)):
    r = json.load(open(f))
    runs += r['runs']
    configs.append({k: v for k, v in r['config'].items() if k not in ('seeds', 'out', 'data')})
assert runs, f'no results.json under {root}'
if any(c != configs[0] for c in configs):
    print('WARNING: runs were made with different configs')

keys = list(runs[0]['overall'])
summary = {k: dict(mean=float(np.mean([r['overall'][k] for r in runs])),
                   std=float(np.std([r['overall'][k] for r in runs]))) for k in keys}
seeds = sorted(r['seed'] for r in runs)

probs, label = [], None
for f in sorted(glob.glob(os.path.join(root, '**', 'predictions_seed*.npz'), recursive=True)):
    d = np.load(f)
    if (d['fold_id'] < 0).any():
        print(f'skipping {f} for the ensemble (not all folds were run)')
        continue
    label = d['label']
    probs.append(1 / (1 + np.exp(-d['test_logits'].astype(np.float64))))
ensemble = None
if len(probs) > 1:
    p = np.mean(probs, axis=0).ravel()
    ensemble = {k: float(v) for k, v in paper_metrics(label.ravel(), p).items()}
    ensemble.update({k: float(v) for k, v in sk_metrics(label.ravel(), p).items()})

json.dump(dict(config=configs[0], seeds=seeds, summary=summary, ensemble=ensemble,
               ensemble_size=len(probs), runs=runs),
          open(os.path.join(root, 'summary.json'), 'w'), indent=1)

lines = [f'# MRLHGNN (corrected): {len(runs)} seeds x 5-fold CV', '', f'seeds: {seeds}', '',
         f'config: `{json.dumps(configs[0])}`', '',
         '| metric | mean | std |', '|---|---|---|']
if ensemble:
    lines[-2:] = ['| metric | mean | std | ensemble of %d seeds |' % len(probs), '|---|---|---|---|']
    lines += [f'| {k} | {v["mean"]:.4f} | {v["std"]:.4f} | {ensemble[k]:.4f} |' for k, v in summary.items()]
else:
    lines += [f'| {k} | {v["mean"]:.4f} | {v["std"]:.4f} |' for k, v in summary.items()]
lines += ['', '| seed | ' + ' | '.join(keys) + ' |', '|---' * (len(keys) + 1) + '|']
lines += [f'| {r["seed"]} | ' + ' | '.join(f'{r["overall"][k]:.4f}' for k in keys) + ' |'
          for r in sorted(runs, key=lambda r: r['seed'])]
open(os.path.join(root, 'RESULTS.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
