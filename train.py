"""5-fold cross-validation of MRLHGNN on the benchmark dataset, with correct evaluation.

Protocol (Sec. IV-C of the paper, plus the corrections in reproduction/AUDIT.md):
  * positives and negatives are each split into 5 folds; each drug-disease pair is tested exactly once
  * test positives are removed from the graph; only the test fold's own pairs are scored
  * 10% of each training fold (stratified) is held out as validation: early stopping and model
    selection use validation AUPR only; validation positives are also removed from the graph
  * repeated over several seeds, reported as mean +- std

The defaults are the final configuration used for the reported results (runs/final).

Usage:
  python train.py --out runs/final                      # seeds 0-4, 5-fold CV
  python train.py --seeds 0 --max_folds 1 --epochs 50   # quick check
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold, train_test_split

from data import (DISEASE_METAPATHS, DRUG_METAPATHS, N_DISEASE, N_DRUG, build_adjacency,
                  load_dataset, metapath_operators)
from metrics import paper_metrics, sk_metrics
from model import MRLHGNN

HERE = os.path.dirname(os.path.abspath(__file__))


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data', default=os.path.join(HERE, 'data'))
    p.add_argument('--out', default=os.path.join(HERE, 'runs', 'default'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--max_folds', type=int, default=None, help='stop after this many folds (quick tests)')
    # as in Sec. IV-A of the paper
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.4)
    p.add_argument('--topk', type=int, default=15)
    # selected on validation AUPR (paper: lr 0.005, loss weight |S-|/|S+| ~ 150)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--pos_weight', type=float, default=50.0,
                   help='positive-class weight in the loss (Eq. 12); <= 0 uses |S-|/|S+| as in the paper')
    # training settings not given in the paper
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--epochs', type=int, default=2500, help='maximum epochs')
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--eval_every', type=int, default=10)
    p.add_argument('--patience', type=int, default=200, help='epochs without val-AUPR improvement')
    # ablations (Fig. 5)
    p.add_argument('--no_layer_norm', action='store_true')
    p.add_argument('--no_transformer', action='store_true')
    p.add_argument('--no_decoder_attention', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pairs_to_mask(pairs):
    m = np.zeros((N_DRUG, N_DISEASE), bool)
    m[pairs[:, 0], pairs[:, 1]] = True
    return m


def run_fold(args, ds, train_pos, train_neg, seed, log):
    dev = torch.device(args.device)
    # stratified train/validation split inside the training fold
    tr_pos, va_pos = train_test_split(train_pos, test_size=args.val_frac, random_state=seed)
    tr_neg, va_neg = train_test_split(train_neg, test_size=args.val_frac, random_state=seed)
    tr_mask = pairs_to_mask(np.vstack([tr_pos, tr_neg]))
    va_mask = pairs_to_mask(np.vstack([va_pos, va_neg]))

    adj = build_adjacency(ds, tr_pos)  # graph only contains training positives
    ops_r = metapath_operators(adj, DRUG_METAPATHS, dev)
    ops_d = metapath_operators(adj, DISEASE_METAPATHS, dev)
    x_r = torch.tensor(ds['x_drug'], device=dev)
    x_d = torch.tensor(ds['x_dis'], device=dev)
    y = torch.tensor(ds['label'], device=dev)
    tr_m, va_m = torch.tensor(tr_mask, device=dev), torch.tensor(va_mask, device=dev)
    y_va = ds['label'][va_mask]

    model = MRLHGNN(x_r.shape[1], N_DRUG, N_DISEASE, len(DRUG_METAPATHS), args.hidden, args.dropout,
                    layer_norm=not args.no_layer_norm, use_transformer=not args.no_transformer,
                    decoder_attention=not args.no_decoder_attention).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=0.1 * args.lr, max_lr=args.lr, gamma=0.995,
                                              step_size_up=20, mode='exp_range', cycle_momentum=False)
    # Eq. (12): weighted cross-entropy; the paper uses lambda = |S-| / |S+|
    pos_weight = args.pos_weight if args.pos_weight > 0 else len(tr_neg) / len(tr_pos)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))

    best = (-1.0, 0, None)  # (val AUPR, epoch, state)
    for ep in range(1, args.epochs + 1):
        model.train()
        logits = model(x_r, x_d, ops_r, ops_d)
        loss = crit(logits[tr_m], y[tr_m])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                s = torch.sigmoid(model(x_r, x_d, ops_r, ops_d))[va_m].cpu().numpy()
            ap = average_precision_score(y_va, s)
            if ap > best[0]:
                best = (ap, ep, copy.deepcopy(model.state_dict()))
            if ep % 100 == 0:
                log(f'    ep {ep:4d} loss {loss.item():.4f} val AUPR {ap:.4f} (best {best[0]:.4f} @ {best[1]})')
            if ep - best[1] >= args.patience:
                break

    model.load_state_dict(best[2]); model.eval()
    with torch.no_grad():
        logits = model(x_r, x_d, ops_r, ops_d).cpu().numpy()
    return dict(logits=logits, val_mask=va_mask, best_epoch=best[1], best_val_aupr=best[0],
                view_weights=model.view_weights().detach().flatten().cpu().numpy())


def run_seed(args, ds, seed, log):
    set_seed(seed)
    Y = ds['label']
    pos = np.argwhere(Y == 1); neg = np.argwhere(Y == 0)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=seed)
    test_logits = np.full(Y.shape, np.nan, np.float32)
    fold_id = np.full(Y.shape, -1, np.int8)
    val_records, fold_metrics = [], []
    for f, ((trp, tep), (trn, ten)) in enumerate(zip(kf.split(pos), kf.split(neg))):
        if args.max_folds is not None and f >= args.max_folds:
            break
        t0 = time.time()
        test_mask = pairs_to_mask(np.vstack([pos[tep], neg[ten]]))
        r = run_fold(args, ds, pos[trp], neg[trn], seed * 100 + f, log)
        test_logits[test_mask] = r['logits'][test_mask]  # only this fold's test pairs
        fold_id[test_mask] = f
        vi = np.argwhere(r['val_mask'])
        val_records.append(np.column_stack([np.full(len(vi), f), vi, Y[r['val_mask']], r['logits'][r['val_mask']]]))
        p = 1 / (1 + np.exp(-r['logits'][test_mask]))
        fm = {k: float(v) for k, v in paper_metrics(Y[test_mask], p).items()}
        fm.update(best_epoch=r['best_epoch'], best_val_aupr=float(r['best_val_aupr']),
                  view_weights=[round(float(w), 4) for w in r['view_weights']])
        fold_metrics.append(fm)
        log(f'  seed {seed} fold {f + 1}: AUC {fm["AUC"]:.4f} AUPR {fm["AUPR"]:.4f} F1 {fm["F1"]:.4f} '
            f'(best ep {r["best_epoch"]}, {time.time() - t0:.0f}s)')

    done = fold_id >= 0
    prob = 1 / (1 + np.exp(-test_logits[done]))
    overall = {k: float(v) for k, v in paper_metrics(Y[done], prob).items()}
    overall.update({k: float(v) for k, v in sk_metrics(Y[done], prob).items()})
    os.makedirs(args.out, exist_ok=True)
    # saved for calibration / conformal prediction: test logits per pair + validation (calibration) logits
    np.savez_compressed(os.path.join(args.out, f'predictions_seed{seed}.npz'), test_logits=test_logits,
                        fold_id=fold_id, label=Y, val_records=np.vstack(val_records))
    return overall, fold_metrics


def summarize(results):
    keys = [k for k in results[0]['overall']]
    return {k: dict(mean=float(np.mean([r['overall'][k] for r in results])),
                    std=float(np.std([r['overall'][k] for r in results]))) for k in keys}


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, 'train.log'), 'a', encoding='utf-8')

    def log(msg):
        print(msg, flush=True); logf.write(msg + '\n'); logf.flush()

    log(f'config: {json.dumps(vars(args))}')
    ds = load_dataset(args.data, args.topk)
    results = []
    for seed in args.seeds:
        t0 = time.time()
        overall, folds = run_seed(args, ds, seed, log)
        results.append(dict(seed=seed, overall=overall, folds=folds))
        log(f'seed {seed}: ' + ' '.join(f'{k} {v:.4f}' for k, v in overall.items()) +
            f' ({(time.time() - t0) / 60:.1f} min)')
        summary = summarize(results)
        with open(os.path.join(args.out, 'results.json'), 'w') as fh:
            json.dump(dict(config=vars(args), summary=summary, runs=results), fh, indent=1)
    log('SUMMARY (mean +- std over seeds): ' +
        ' | '.join(f'{k} {v["mean"]:.4f}+-{v["std"]:.4f}' for k, v in summary.items()))


if __name__ == '__main__':
    main()
