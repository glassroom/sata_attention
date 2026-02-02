import warnings
import math
import torch
from itertools import combinations_with_replacement


def _calculate_n_idx_permutations(M: torch.Tensor, d_key: int) -> torch.Tensor:
    """
    Calculate the number of multiset permutations of each row of M. See
    https://en.wikipedia.org/wiki/Permutation#Permutations_of_multisets.
    """
    log_factorial = lambda counts: torch.lgamma(counts + 1)
    vmap_bincount = torch.vmap(lambda row: torch.bincount(row, minlength=d_key))
    with warnings.catch_warnings(action="ignore"):
        bin_counts = vmap_bincount(M)
    log_numer = log_factorial(torch.tensor(M.size(-1)))
    log_denoms = torch.sum(log_factorial(bin_counts), dim=-1)
    return torch.exp(log_numer - log_denoms).round().long()


class TightlyPackedTaylorTerm(torch.nn.Module):
    """
    Proof-of-concept implementation of tightly packed Taylor numerator and
    denominator terms, as proposed "Self-Attention at Constant Cost per Token
    via Symmetry-Aware Taylor Approximation" (Heinsen and Kozachkov, 2026).

    Args:
        d_key: int, number of elements per head in queries and keys.
        d_val: int, number of elements per head in values.
        p: int, power of Taylor term (order of symmetric tensor products).
        is_causal: bool. If True, computes autoregressive attention.

    Inputs:
        Q: float tensor of shape [..., n_qry, d_key].
        K: float tensor of shape [..., n_tok, d_key].
        V: float tensor of shape [..., n_tok, d_val].
        continue_prev: bool, if True, continues the sequence.

    Outputs:
        S_term: float tensor of shape [..., n_qry, d_val].
        Z_term: float tensor of shape [..., n_qry, 1].
    """
    def __init__(self, d_key: int, d_val: int, p: int, is_causal: bool) -> None:
        super().__init__()
        self.d_key, self.d_val, self.p, self.is_causal = (d_key, d_val, p, is_causal)
        self.register_buffer('alpha', torch.tensor(1.0 / (math.factorial(p) * (d_key ** (p / 2)))))
        self.register_buffer('M', torch.tensor([*combinations_with_replacement(range(d_key), p)], dtype=torch.long))
        self.register_buffer('C', _calculate_n_idx_permutations(self.M, d_key).float())
        assert len(self.M) == math.comb(d_key + p - 1, p) and self.C.long().sum() == d_key ** p  # verify init

    def acummulate(self, summands: torch.Tensor) -> torch.Tensor:
        if self.is_causal:
            return torch.cumsum(summands, dim=-3)             # [..., n, :, :] -> [..., n, :, :]
        else:
            return torch.sum(summands, dim=-3, keepdim=True)  # [..., n, :, :] -> [..., 1, :, :]

    def Phi(self, x):
        return x[..., self.M].prod(dim=-1)  # note: x[..., self.M] returns *copies*, not views

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, continue_prev: bool) -> tuple[torch.Tensor, torch.Tensor]:
        assert Q.size(-1) == K.size(-1) and K.size(-2) == V.size(-2), (
            "Input shapes are incompatible. See docstring for required input shapes.")
        if self.is_causal:
            assert Q.size(-2) == K.size(-2) == V.size(-2), (
                "Number of queries, keys, and values must match for causal attention.")

        if self.p == 0:
            H_S_summands = V[..., None, :]                                   # [..., n_tok, m, d_val], m is 1
            H_Z_summands = torch.ones_like(V[..., None, :1])                 # [..., n_tok, m, 1],     m is 1

            H_S = self.acummulate(H_S_summands)                              # [..., (n_qry=n_tok if causal, else 1), m, d_val]
            H_Z = self.acummulate(H_Z_summands)                              # [..., (n_qry=n_tok if causal, else 1), m, 1]

            if continue_prev:
                H_S = self.prev_H_S + H_S                                    # [..., (n_qry or 1), m, d_val]
                H_Z = self.prev_H_Z + H_Z                                    # [..., (n_qry or 1), m, 1]

            S_term = H_S.squeeze(-2).expand(*Q.shape[:-1], -1)               # [..., n_qry, d_val]
            Z_term = H_Z.squeeze(-2).expand(*Q.shape[:-1], -1)               # [..., n_qry, 1]
        else:
            Phi_Q = self.Phi(Q)                                              # [..., n_qry, m]
            Phi_K = self.Phi(K)                                              # [..., n_tok, m]

            H_S_summands = Phi_K[..., None] * V[..., None, :]                # [..., n_tok, m, d_val]
            H_Z_summands = Phi_K[..., None]                                  # [..., n_tok, m, 1]
    
            H_S = self.acummulate(H_S_summands) * self.alpha                 # [..., (n_qry=n_tok if causal, else 1), m, d_val]
            H_Z = self.acummulate(H_Z_summands) * self.alpha                 # [..., (n_qry=n_tok if causal, else 1), m, 1]
    
            if continue_prev:
                H_S = self.prev_H_S + H_S                                    # [..., (n_qry or 1), m, d_val]
                H_Z = self.prev_H_Z + H_Z                                    # [..., (n_qry or 1), m, 1]

            S_term = torch.einsum('m,...m,...md->...d', self.C, Phi_Q, H_S)  # [..., n_qry, d_val]
            Z_term = torch.einsum('m,...m,...md->...d', self.C, Phi_Q, H_Z)  # [..., n_qry, 1]
    
        self.prev_H_S = H_S[..., -1:, :, :].detach()                         # [..., 1, m, d_val]
        self.prev_H_Z = H_Z[..., -1:, :, :].detach()                         # [..., 1, m, 1]

        return S_term, Z_term


class SymmetryAwareTaylorApproximatedAttention(torch.nn.Module):
    """
    Proof-of-concept implementation of "Self-Attention at Constant Cost per
    Token via Symmetry-Aware Taylor Approximation" (Heinsen and Kozachkov, 2026).

    Args:
        d_key: int, number of elements per head in queries and keys.
        d_val: int, number of elements per head in values.
        is_causal: bool. If True, computes autoregressive attention.
        n_taylor: optional int. Number of Taylor terms. Default: 4.

    Inputs:
        Q: float tensor of shape [..., n_qry, d_key] with queries.
        K: float tensor of shape [..., n_tok, d_key] with keys.
        V: float tensor of shape [..., n_tok, d_val] with values.
        continue_prev: bool, if True, continues the sequence.

    Output:
        Y: float tensor of shape [..., n_qry, d_val] with attention.
    """
    def __init__(self, d_key: int, d_val: int, is_causal: bool, n_taylor: int=4) -> None:
        super().__init__()
        self.d_key, self.d_val, self.is_causal, self.n_taylor = (d_key, d_val, is_causal, n_taylor)
        self.tptts = torch.nn.ModuleList([TightlyPackedTaylorTerm(d_key, d_val, p, is_causal) for p in range(n_taylor)])

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V:torch. Tensor, continue_prev: bool=False) -> torch.Tensor:
        iter_over_tptts = (tptt(Q, K, V, continue_prev) for tptt in self.tptts)  # note: can be parallelized
        S_terms, Z_terms = zip(*iter_over_tptts)
        S = torch.stack(S_terms, dim=0).sum(dim=0)
        Z = torch.stack(Z_terms, dim=0).sum(dim=0)
        Y = torch.nan_to_num(S / Z)
        return Y

    # Convenience methods:

    def get_forward_FLOPs_per_query_head(self) -> dict:
        fpt = { f'tptt[{i}]': (2 + 2 * (tptt.p + 1) + 4 * tptt.d_val) * tptt.C.numel() for i, tptt in enumerate(self.tptts) }
        fpt['Total'] = sum(v for k, v in fpt.items())
        return fpt

    def get_hidden_state_sizes(self) -> dict:
        szs = { f'tptt[{i}]': math.comb(tptt.d_key + tptt.p - 1, tptt.p) * (tptt.d_val + 1) for i, tptt in enumerate(self.tptts) }
        szs['Total'] = sum(v for k, v in szs.items())
        return szs
