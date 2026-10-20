"""Benchmark dataset loading and heterogeneous-graph construction for MRLHGNN.

Replaces the DGL graph of the original code with scipy sparse adjacency matrices, so it
runs on any OS / PyTorch version. Node types: drug (r), disease (d), protein (p), pathway (a).
"""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize

N_DRUG, N_DISEASE = 894, 454

# View-specific metapaths, Table III of the paper (GP^1 ... GP^8)
DRUG_METAPATHS = [
    ['drug_drug'],                                                      # R-R
    ['drug_protein', 'protein_drug'],                                   # R-P-R
    ['drug_disease', 'disease_drug'],                                   # R-D-R
    ['drug_drug', 'drug_protein', 'protein_drug'],                      # R-R-P-R
    ['drug_drug', 'drug_disease', 'disease_drug'],                      # R-R-D-R
    ['drug_protein', 'protein_drug', 'drug_drug'],                      # R-P-R-R
    ['drug_disease', 'disease_drug', 'drug_drug'],                      # R-D-R-R
    ['drug_disease', 'disease_pathway', 'pathway_disease', 'disease_drug'],  # R-D-A-D-R
]
DISEASE_METAPATHS = [
    ['disease_disease'],                                                # D-D
    ['disease_pathway', 'pathway_disease'],                             # D-A-D
    ['disease_drug', 'drug_disease'],                                   # D-R-D
    ['disease_disease', 'disease_pathway', 'pathway_disease'],          # D-D-A-D
    ['disease_disease', 'disease_drug', 'drug_disease'],                # D-D-R-D
    ['disease_pathway', 'pathway_disease', 'disease_disease'],          # D-A-D-D
    ['disease_drug', 'drug_disease', 'disease_disease'],                # D-R-D-D
    ['disease_drug', 'drug_protein', 'protein_drug', 'drug_disease'],   # D-R-P-R-D
]


def topk_edges(sim, k):
    """Binarise a similarity matrix: each row keeps its k most similar entries (Sec. III-A).

    The original code used np.argpartition(row, k)[-k:], which does not select the top-k.
    """
    idx = np.argpartition(sim, -k, axis=1)[:, -k:]
    rows = np.repeat(np.arange(len(sim)), k)
    return rows, idx.ravel()


def load_dataset(data_dir, topk=15):
    rd = lambda f, **kw: pd.read_csv(os.path.join(data_dir, f), **kw)
    label = rd('KFCdataset_baseline.csv', header=None).values.astype(np.float32)
    drug_sim = rd('drug_drug_baseline.csv', header=None).values.astype(np.float32)
    dis_sim = rd('disease_disease_baseline.csv', header=None).values.astype(np.float32)
    pp = rd('protein_protein.csv')
    aa = rd('pathway_pathway.csv')
    rp = rd('drug_protein.csv')
    ad = rd('pathway_disease.csv')
    assert label.shape == (N_DRUG, N_DISEASE)

    n = {'drug': N_DRUG, 'disease': N_DISEASE,
         'protein': int(max(rp.Protein.max(), pp.Protein1.max(), pp.Protein2.max())) + 1,
         'pathway': int(max(aa.Pathway1.max(), aa.Pathway2.max(), ad.Pathway.max())) + 1}
    rr, ss = topk_edges(drug_sim, topk), topk_edges(dis_sim, topk)
    # drug-disease edges are added per fold from the training labels only
    edges = {
        ('drug', 'drug_drug', 'drug'): rr,
        ('disease', 'disease_disease', 'disease'): ss,
        ('drug', 'drug_protein', 'protein'): (rp.Drug.values, rp.Protein.values),
        ('protein', 'protein_drug', 'drug'): (rp.Protein.values, rp.Drug.values),
        ('protein', 'protein_protein', 'protein'): (pp.Protein1.values, pp.Protein2.values),
        ('pathway', 'pathway_pathway', 'pathway'): (aa.Pathway1.values, aa.Pathway2.values),
        ('pathway', 'pathway_disease', 'disease'): (ad.Pathway.values, ad.Disease.values),
        ('disease', 'disease_pathway', 'pathway'): (ad.Disease.values, ad.Pathway.values),
    }
    # Eq. (2): X = [[S_r, 0], [0, S_d]]
    x_drug = np.hstack([drug_sim, np.zeros((N_DRUG, N_DISEASE), np.float32)])
    x_dis = np.hstack([np.zeros((N_DISEASE, N_DRUG), np.float32), dis_sim])
    return dict(label=label, edges=edges, n=n, x_drug=x_drug, x_dis=x_dis)


def build_adjacency(ds, known_pairs):
    """Per-relation adjacency (src x dst, csr). Drug-disease edges = `known_pairs` (training positives)."""
    n = ds['n']
    adj = {}
    for (s, e, d), (u, v) in ds['edges'].items():
        adj[e] = sp.coo_matrix((np.ones(len(u), np.float32), (u, v)), shape=(n[s], n[d])).tocsr()
    rd = sp.coo_matrix((np.ones(len(known_pairs), np.float32), (known_pairs[:, 0], known_pairs[:, 1])),
                       shape=(N_DRUG, N_DISEASE)).tocsr()
    adj['drug_disease'], adj['disease_drug'] = rd, rd.T.tocsr()
    return adj


def metapath_operators(adj, metapaths, device):
    """Eq. (4): product of row-normalised adjacency matrices along each metapath, as dense tensors."""
    ops = []
    for mp in metapaths:
        prod = None
        for e in mp:
            m = normalize(adj[e], norm='l1', axis=1)
            prod = m if prod is None else prod @ m
        ops.append(torch.tensor(prod.toarray(), dtype=torch.float32, device=device))
    return ops
