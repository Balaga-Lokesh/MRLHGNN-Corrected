"""MRLHGNN model (Peng et al., IEEE JBHI 2025), with the fixes from reproduction/AUDIT.md.

Pipeline per node type (drug / disease), M metapath views + 1 raw-feature view:
  1. metapath-level aggregation   X_GP = Y_hat(...)Y_hat (w * X)            Eq. (4), w trainable
  2. view-specific projection     H'_GP = MLP_GP(X_GP)                       Eq. (5), with LayerNorm
  3. transformer view fusion      h = phi * sum_j alpha_ij v_j + h'_i        Eq. (6)-(10)
  4. auto multi-view decoder      Y = sum_n beta_n H_r^n W_n (H_d^n)^T      Eq. (11)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def projector(in_dim, hidden, dropout, layer_norm):
    mid = (in_dim + hidden) // 2

    def block(a, b):
        return [nn.Linear(a, b)] + ([nn.LayerNorm(b)] if layer_norm else []) + [nn.ReLU(), nn.Dropout(dropout)]

    return nn.Sequential(*block(in_dim, mid), *block(mid, mid), nn.Linear(mid, hidden))


class ViewEncoder(nn.Module):
    """Encodes one node type into (M+1) view-specific embeddings."""

    def __init__(self, n_nodes, in_dim, hidden, n_metapaths, dropout, layer_norm, use_transformer):
        super().__init__()
        self.n_views = n_metapaths + 1
        self.w = nn.Parameter(torch.empty(n_metapaths, n_nodes, 1).uniform_())
        self.proj = nn.ModuleList([projector(in_dim, hidden, dropout, layer_norm) for _ in range(self.n_views)])
        self.q, self.k, self.v = (nn.Linear(hidden, hidden) for _ in range(3))
        self.phi = nn.Parameter(torch.ones(1))
        self.use_transformer = use_transformer

    def forward(self, x, ops):
        feats = [x] + [P @ (w * x) for P, w in zip(ops, self.w)]
        h = torch.stack([p(f) for p, f in zip(self.proj, feats)], dim=1)   # (N, V, H)
        if self.use_transformer:
            attn = torch.softmax(self.q(h) @ self.k(h).transpose(1, 2), dim=-1)  # (N, V, V)
            h = self.phi * (attn @ self.v(h)) + h
        return h.transpose(0, 1)                                            # (V, N, H)


class MRLHGNN(nn.Module):
    def __init__(self, in_dim, n_drug, n_disease, n_metapaths, hidden=128, dropout=0.4,
                 layer_norm=True, use_transformer=True, decoder_attention=True):
        super().__init__()
        args = (in_dim, hidden, n_metapaths, dropout, layer_norm, use_transformer)
        self.drug_enc = ViewEncoder(n_drug, *args)
        self.dis_enc = ViewEncoder(n_disease, *args)
        n_views = n_metapaths + 1
        self.dec_W = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(n_views)])
        self.beta = nn.Parameter(torch.zeros(n_views, 1, 1))
        self.decoder_attention = decoder_attention

    def view_weights(self):
        if self.decoder_attention:
            return F.softmax(self.beta, dim=0)
        return torch.full_like(self.beta, 1.0 / len(self.beta))

    def forward(self, x_drug, x_dis, ops_drug, ops_dis):
        hr = self.drug_enc(x_drug, ops_drug)
        hd = self.dis_enc(x_dis, ops_dis)
        scores = torch.stack([r @ W(d).T for r, W, d in zip(hr, self.dec_W, hd)])  # (V, Nr, Nd)
        return (scores * self.view_weights()).sum(0)                              # logits
