# MRLHGNN: corrected implementation

This is our implementation of **MRLHGNN** from Peng et al., "Drug Repositioning via Multi-View Representation Learning With
Heterogeneous Graph Neural Network", *IEEE JBHI* 29(3), 2025 (doi:10.1109/JBHI.2024.3434439). It follows the paper's
architecture and fixes the defects we found in the authors' released code
([biohnuster/MRLHGNN](https://github.com/biohnuster/MRLHGNN)), which are listed in §1. The most important one is an
evaluation bug that inflates the reported results.

## 1. Method

The task is link prediction on a heterogeneous graph: score every drug–disease pair.

**Heterogeneous graph** (`data.py`). Four node types: drug (R), disease (D), protein (P) and pathway (A). Edges:
- **R–R and D–D:** each node is linked to its 15 most similar nodes. Drug similarity is the Tanimoto similarity of
  molecular fingerprints; disease similarity is MeSH semantic similarity. Self-links are included.
- **Known biological links:** R–P, P–P, A–A and D–A.
- **R–D:** the known drug–disease associations in the current *training* split only.

Each relation is stored as a sparse adjacency matrix (scipy); no DGL is needed.

**Node features** (Eq. 2). A drug's feature vector is its row of the drug-similarity matrix padded with zeros, and the same for
diseases: X = [[S_r, 0], [0, S_d]], dimension 894 + 454 = 1348.

**Model** (`model.py`). It is applied separately to drugs and to diseases, with M = 8 metapath views plus 1 raw-feature view:

1. **Metapath-level aggregation (Eq. 4).** For each view-specific metapath (Table III, listed below),
   X_GP = Â₁Â₂…Â_l (w ⊙ X). Â is the row-normalised adjacency of each relation on the path, and w is a trainable
   per-node weight vector. The metapath products are computed once per fold, as dense matrices.
2. **View-specific projection (Eq. 5).** Each view has its own MLP: Linear → LayerNorm → ReLU → Dropout, twice,
   then Linear → 128 dimensions.
3. **Transformer view fusion (Eq. 6–10).** For each node, attention runs across its 9 view embeddings:
   q, k, v = W_Q h, W_K h, W_V h; α = softmax(q kᵀ); h ← φ·Σ α v + h. W_Q, W_K, W_V and φ are shared across views.
4. **Auto multi-view fusion decoder (Eq. 11).** Ŷ = Σₙ βₙ · H_r⁽ⁿ⁾ Wₙ (H_d⁽ⁿ⁾)ᵀ, with β = softmax over learned view
   weights. Ŷ holds the 894 × 454 association logits.

| View | Drug metapath | Disease metapath |
|---|---|---|
| 0 | raw features | raw features |
| 1 | R→R | D→D |
| 2 | R→P→R | D→A→D |
| 3 | R→D→R | D→R→D |
| 4 | R→R→P→R | D→D→A→D |
| 5 | R→R→D→R | D→D→R→D |
| 6 | R→P→R→R | D→A→D→D |
| 7 | R→D→R→R | D→R→D→D |
| 8 | R→D→A→D→R | D→R→P→R→D |

**Loss (Eq. 12).** Weighted binary cross-entropy over all training pairs, with positive-class weight λ = 50.

Model size: 29.75 M trainable parameters, almost all in the 18 projection MLPs, whose input is 1348-dimensional.

### Differences from the authors' released code
| Component | Released code | This implementation |
|---|---|---|
| Collecting test predictions | `pred[list_idx]` copies whole rows, so training pairs get scored | only each fold's own test pairs are scored |
| Model selection | early stopping on training loss/AUC | early stopping on validation AUPR |
| Top-k similarity graph | `argpartition(row, k)[-k:]` isn't the top-k | real top-15 |
| Node features | overwritten in place by the top-k step | raw similarity (Eq. 2) |
| Transformer output | `.view()` scrambles nodes across views | `.transpose()` |
| Transformer value | V computed but unused | Eq. 10 with V |
| Metapath weights w | detached, never trained | trainable (Eq. 4) |
| Hidden size | 64 | 128 (paper) |
| Projection MLP | no normalization | LayerNorm (paper) |

## 2. Experimental setup

**Dataset.** The paper's benchmark dataset, copied from the authors' repository into `data/`:

| Item | Count |
|---|---|
| Drugs / diseases / proteins / pathways | 894 / 454 / 18,877 / 314 |
| Known drug–disease associations (positives) | 2,704 |
| Unknown pairs (treated as negatives) | 403,172 (negatives : positives ≈ 149 : 1, 0.67% positive) |
| Drug–drug edges (top-15) / disease–disease edges (top-15) | 13,410 / 6,810 |
| Drug–protein / protein–protein / pathway–pathway / disease–pathway | 4,397 / 2,013,782 / 1,669 / 19,530 |

**Evaluation protocol** (`train.py`):
- **5-fold cross-validation** (paper, Sec. IV-C). Positives and negatives are each split into 5 folds. In each round, 4 folds
  train and 1 fold tests, so every one of the 405,876 pairs is tested exactly once.
- **Per fold:** about 2,163 training and 541 test positives; about 322,538 training and 80,634 test negatives.
- **Validation split:** 10% of each training fold (stratified, about 216 positives and 32,254 negatives) is held out.
  Early stopping and checkpoint selection use **validation AUPR**. The test fold is never used for any decision.
- **No leakage:** test *and* validation positives are removed from the graph before the metapath matrices are built,
  so the model can't see them through R–D edges.
- **Repeats:** the whole procedure runs with **5 seeds (0–4)**. Each seed reshuffles the folds and re-initialises the model.
  We report mean ± std over seeds.
- **Seed ensemble:** we also average the 5 seeds' held-out probabilities per pair. Each seed's prediction for a pair
  comes from a model that never trained on that pair, so this is still a held-out evaluation.

**Metrics** (`metrics.py`, same procedure as the paper, from LAGCN):
- **AUC** and **AUPR**: trapezoid areas over 999 score thresholds, on all 405,876 out-of-fold predictions.
- **F1, Accuracy, Recall, Specificity, Precision**: at the threshold that maximises F1.
- sklearn's ROC-AUC and average precision are also stored, for transparency.

**Hyper-parameters:**

| Parameter | Value | Source |
|---|---|---|
| Hidden dimension | 128 | paper |
| Dropout | 0.4 | paper |
| Top-k similarity neighbours | 15 | paper |
| Optimizer | Adam, weight decay 0 | paper / released code |
| Learning rate | 0.001 | selected on validation (paper: 0.005) |
| LR schedule | CyclicLR exp_range, base 0.1·lr, max lr, step 20, γ 0.995 | released code |
| Positive-class loss weight λ | 50 | selected on validation (paper: \|S⁻\|/\|S⁺\| ≈ 149) |
| Max epochs / patience / validation interval | 2500 / 200 / 10 | ours (full-batch training) |
| Validation fraction | 10% of the training fold | ours |
| Seeds | 0, 1, 2, 3, 4 | ours |

**How lr and λ were chosen.** We trained candidate settings on seed 0, folds 1–2, and picked the one with the best
*validation* AUPR. Results:
- **lr:** at the paper's 0.005 the model overfitted within a few hundred epochs. Validation AUPR went from 0.13 to 0.35 at lr 0.001.
- **λ:** lowering it to 50 gave a further +0.03 on validation.
- **Didn't help at lr 0.001:** dropout 0.6, weight decay 1e-4, lr 0.0005, λ 10, hidden size 64/256, top-k 10/25 and
  identity-free features.

**Implementation.**
- **Software:** Python 3.12.10, PyTorch 2.11.0 (CUDA 12.8), NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6.
- **Hardware:** Windows 11, NVIDIA RTX 5070 Laptop GPU (8 GB), AMD Ryzen 9 8940HX, 24 GB RAM.
- **Runtime:** training is full-batch, with all 894 × 454 pairs scored per epoch. One fold takes 49–211 s (median 127 s), measured
  with 3 runs sharing the GPU. The best epoch ranges from 200 to 1710 (median 1130).

## 3. Results

| Method | AUC | AUPR | F1 | Accuracy | Recall | Specificity | Precision |
|---|---|---|---|---|---|---|---|
| Paper, Table IV (reported) | 0.9585 ± 0.0025 | 0.6645 ± 0.0118 | 0.7200 ± 0.0004 | 0.9959 ± 0.0030 | 0.7818 ± 0.0008 | 0.9974 ± 0.0005 | 0.6673 ± 0.0010 |
| Authors' code, their evaluation (buggy)¹ | 0.9593 | 0.6669 | 0.7292 | 0.9962 | 0.7777 | 0.9976 | 0.6864 |
| Authors' code, correct evaluation¹ | 0.7804 | 0.1014 | 0.1505 | 0.9914 | 0.1139 | 0.9973 | 0.2219 |
| **Ours** (5 seeds, mean ± std) | **0.8071 ± 0.0077** | **0.3496 ± 0.0096** | **0.4220 ± 0.0063** | 0.9932 ± 0.0003 | 0.3702 ± 0.0194 | 0.9974 ± 0.0004 | 0.4939 ± 0.0289 |
| **Ours, 5-seed ensemble** | **0.8460** | **0.4262** | **0.4740** | 0.9939 | 0.4149 | 0.9977 | 0.5527 |

¹ The authors' released code run unchanged, seed 5800. The only difference between the two rows is
how test predictions are collected.

Per-seed numbers are in [`runs/final/RESULTS.md`](runs/final/RESULTS.md). Accuracy and specificity are near 1 for every
method because 99.3% of pairs are negatives, so AUC, AUPR and F1 are the informative metrics.

## 4. Reproducing

```bash
pip install -r requirements.txt
python train.py --seeds 0 --max_folds 1 --epochs 50 --out runs/smoke     # ~30 s sanity check
python train.py --out runs/final                                          # full run: 5 seeds x 5 folds, about 1 h
```
The defaults are the configuration above. To spread the seeds over several terminals, give each its own folder and
merge them afterwards:
```bash
python train.py --seeds 0 1 --out runs/final/a
python train.py --seeds 2 3 --out runs/final/b
python train.py --seeds 4   --out runs/final/c
python aggregate.py runs/final        # mean ± std over seeds + seed ensemble -> RESULTS.md, summary.json
```
Paper-component ablations: `--no_transformer`, `--no_decoder_attention`, `--no_layer_norm`. To use the paper's loss
weight, pass `--pos_weight 0`.

## 5. Files

| File | Contents |
|---|---|
| `data.py` | dataset loading, top-k similarity graphs, adjacency matrices, metapath operators (Table III) |
| `model.py` | MRLHGNN: metapath aggregation, projection MLPs, transformer fusion, multi-view decoder (Eq. 4–11) |
| `train.py` | cross-validation, validation split and early stopping, metrics, prediction export |
| `metrics.py` | the paper's metric procedure (999 thresholds, F1-optimal threshold) + sklearn AUC/AP |
| `aggregate.py` | merges runs from several terminals; mean ± std over seeds and the seed ensemble |
| `data/` | benchmark dataset (from the authors' repository) |
| `runs/final/` | the reported run, with `train.log`, `results.json` and `predictions_seed*.npz` per run folder |

Each `predictions_seed{S}.npz` contains:
- `test_logits` (894 × 454): out-of-fold logits
- `fold_id`, `label`
- `val_records` rows of `[fold, drug, disease, label, logit]`
