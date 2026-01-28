# sata_attention

Proof-of-concept implementation of symmetry-aware Taylor-approximated attention, as proposed in "Self-Attention at Constant Cost per Token via Symmetry-Aware Taylor Expansion" (Heinsen and Kozachkov, 2026).

We show that scaled dot-product attention is efficiently computable to arbitrary precision at constant cost per token, achieving orders-of-magnitude reductions in memory use and computation.

Our work enables unbounded token generation at modest fixed cost, opening a new avenue for reducing the infrastructure and energy demands of large-scale Transformer models.


## Key Insight

The core operation of scaled dot-product attention is exponentiation of a query-key dot-product. Given query and key vectors $q, k \in \mathbb{R}^{d_K}$ and a conventional constant $c = \sqrt{d_K}$, the core operation's Taylor expansion is:

$$
\begin{aligned}
\exp \left( \frac{q^\top k}{c} \right)
& = 1 + \frac{q^\top k}{c} + \frac{1}{2!} \left( \frac{q^\top k}{c} \right)^2 + \frac{1}{3!} \left( \frac{q^\top k}{c} \right)^3 + \dots \\
\\
& = \sum_{p=0}^{\infty}  \alpha_p  (q^\top k)^p, \qquad \alpha_p := \frac{1}{p! c^p} \\
\end{aligned}
$$

Previous efforts to approximate attention via Taylor expansion have stopped at the quadratic term (_i.e._, second order) due to the perceived complexity of evaluating all necessary polynomial interactions for higher-degree terms. As we show in our paper, each term in the Taylor expansion decomposes into an expression over symmetric chains of tensor products. For example, $(q^\top k)^3$ decomposes as:

$$
\begin{aligned}
(q^\top k)^3
& = \sum_{i_1=1}^{d_K} q_{i_1} k_{i_1} \sum_{i_2=1}^{d_K} q_{i_2} k_{i_2} \sum_{i_3=1}^{d_K} q_{i_3} k_{i_3} \\
\\
& = \sum_{i_1=1}^{d_K} \sum_{i_2=1}^{d_K} \sum_{i_3=1}^{d_K} (q_{i_1} q_{i_2} q_{i_3}) (k_{i_1} k_{i_2} k_{i_3}) \\
\\
& = \sum  (q \otimes q \otimes q) \odot (k \otimes k \otimes k) \\
\\
& = \sum \left( q^{\otimes 3} \right) \odot \left( k^{\otimes 3} \right)
\end{aligned}
$$

where $\odot$ denotes elementwise (Hadamard) product and $\otimes$ denotes tensor (outer) product. In code:

```python
import torch

d_key = 4  # toy example

q = torch.randn(4)
k = torch.randn(4)
print((q @ k) ** 3)

q_tensorprod_3_times = torch.einsum('i,j,k->ijk', q, q, q)  # shape is d_key x d_key x d_key
k_tensorprod_3_times = torch.einsum('i,j,k->ijk', k, k, k)  # shape is d_key x d_key x d_key
print(torch.sum(q_tensorprod_3_times * k_tensorprod_3_times))  # same output
```

The tensors $q^{\otimes 3}$ and $k^{\otimes 3}$ are _symmetric_, and their elementwise product, $\left( q^{\otimes 3} \right) \odot \left( k^{\otimes 3} \right) = \left( q \odot k \right)^{\otimes 3}$, is also _symmetric_. As we show in our paper, the upper hyper-triangular region of each of these symmetric tensors contains its unique elements (analogous to the upper triangular region of a symmetric matrix).

By construction, $q^{\otimes 3}$ and $k^{\otimes 3}$ consist of all possible degree-3 monomials of $q$ and $p$, respectively, so their upper hyper-triangular region has the unique monomials that make up the _minimal basis_ for computing $(q^\top k)^3$. All monomials outside that region are permutations of a monomial in the region.

The upper hyper-triangular region of an order-3 tensor is indexed by $i_1 \le i_2 \le i_3$, and consists of $m_3 = \binom{d_K + 3 - 1}{3}$ elements, significantly fewer than $d_K \times d_K \times d_K = {d_K}^3$ elements in the full tensor.

Our key contribution is a maximally succinct, computationally efficient, and embarrassingly parallel feed-forward transformation, shown as `Phi()` below, that maps queries and keys to the minimal basis for computing each Taylor term, tightly packed in a vector, reducing space and time costs by orders of magnitude compared to a naive evaluation over the full symmetric tensors:

```python
from itertools import combinations_with_replacement
from sata_attention import _calculate_n_idx_permutations

p = 3  # order of tensors (degree of polynomials)

# Constants (can be precomputed only once, in advance):
M = torch.tensor([*combinations_with_replacement(range(d_key), p)])  # idxs to unique monoms
C = _calculate_n_idx_permutations(M, d_key)                          # counts in full tensor

def Phi(x): return x[..., M].prod(dim=-1)  # proposed feed-forward transformation

print(torch.sum(Phi(q) * Phi(k) * C))      # same output as before
```

We show in our paper how to apply `Phi()` as a kernel function in a form of linear attention, incurring constant cost per token. Notably, space and time complexity becomes inversely proportional to head size, making it cheaper to apply attention over a larger number of smaller heads.

This repository contains an implementation of attention, approximated via Taylor expansion using our method, along with code for verifying its correctness.


## Proof of Concept

Our implementation is an initial one, and should properly be considered a proof of concept. Unlike implementations of the conventional formulation of attention, which has now benefited from nearly a decade of performance optimization work by a large community of AI researchers and practitioners, our formulation is new and has yet to benefit from comparable long-term work.

Targets for performance optimization include:

**Unnecessary Temporary Copying of Query and Key Elements**: _Currently, we make $m_p \times p$ temporary copies of elements from each query and key vector, instead of pointing to those elements, for mapping the vector to $m_p$ monomial features._ This is because the Python expression `x[..., M]` returns *copies* of `x`'s elements, instead of views. Copying elements increases temporary memory use, and can also saturate memory bandwidth, impacting performance. In principle, copying all that data is unnecessary.

**Absence of Optimizations that Exploit Hierarchical Structure of Symmetric Indices**: _Currently, we do not exploit the structure of the indices stored in  matrix $M_p$._ Each row of $M_p$ contains indices $i_1, i_2, \dots, i_p$ organized hierarchically by $i_1 \le i_2 \le \dots \le i_p$, opening additional opportunities for improving computational efficiency. In principle, it should be possible to exploit hierarchical structure to reduce memory and compute use.

**Sequential Instead of Parallel Evaluation of Taylor Terms**: _Currently, we evaluate Taylor Terms sequentially, instead of in parallel_. We queue them on a single stream on an Nvidia GPU. As we discuss in our paper, Taylor terms can be evaluated in parallel, because they are independent of each other.

**Absence of Common On-Device Optimizations**: _Currently, our focus is on validating the correctness of our formulation, not on developing a high-performance implementation._ A more efficient implementation requires writing low-level on-device code (_e.g._, a fused CUDA kernel for Nvidia GPUs) that carefully handles data, both to avoid unnecessarily making temporary copies of it, and to ensure it is more frequently on faster-access memory (_e.g._, HBM instead of SRAM on Nvidia GPUs) as needed for computation.

**Absence of Additional Performance Optimizations**: _Currently, we have not explored any additional performance optimizations._ They include the possibility of reducing the dimensionality of higher-order feature spaces (say, for order greater than four) by applying conventional techniques, such as dropping basis components with low-magnitude coefficients, finding best-fit lower-rank basis approximations, and obtaining fast approximations of the basis via random sampling or random projections.


## Installation

Clone the repository, or download a single file: `sata_attention.py`.

The only dependency is a recent version of PyTorch.


## Toy Example

```python
import torch
from sata_attention import SymmetryAwareTaylorApproximatedAttention

DEVICE = 'cuda'  # change as needed

n_tok, d_key, d_val = (3, 32, 32)

attn = SymmetryAwareTaylorApproximatedAttention(d_key, d_val, is_causal=True, n_taylor=4)
attn = attn.to(DEVICE)

Q = torch.randn(n_tok, d_key, device=DEVICE)
K = torch.randn(n_tok, d_key, device=DEVICE)
V = torch.randn(n_tok, d_val, device=DEVICE)

Y = attn(Q, K, V)
```

Given a new token, you can compute a attention at constant cost per token with:

```python
new_Q = torch.randn(1, d_key, device=DEVICE)
new_K = torch.randn(1, d_key, device=DEVICE)
new_V = torch.randn(1, d_val, device=DEVICE)

Y = attn(new_Q, new_K, new_V, continue_prev=True)  # constant cost per token
```


## Replicating Our Results

We validate the correctness of formulation by applying our proof-of-concept implementation to sequences with up to 100M tokens. To replicate our results, install the dependencies listed in `requirements.txt` (e.g., `pip install -r requirements.txt`), and run the following from the command line:

```
python replicate_results.py
```

Notes: Tested only on Linux. Requires a GPU with at least 40GB of RAM.