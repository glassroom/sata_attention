# sata_attention

Reference implementation of "[Self-Attention at Constant Cost per Token via Symmetry-Aware Taylor Approximation](https://arxiv.org/abs/2602.00294)" (Heinsen and Kozachkov, 2026).

We show that scaled dot-product attention is efficiently computable to arbitrary precision at constant cost per token, _achieving orders-of-magnitude reductions in memory use and computation compared to the conventional formulation_. Our work enables unbounded token generation at modest fixed cost, for substantially reducing the infrastructure and energy demands of large-scale Transformer models. Toy Example:

```python
import torch            # you must install
from tqdm import tqdm   # you must install
from sata_attention import SymmetryAwareTaylorApproximatedAttention

DEVICE = 'cuda'             # change as needed

n_tok = 1_000_000_000       # 1B tokens, no problem!
n_head, d_head = (1024, 8)  # 1K small heads per token, no problem!
n_taylor = 8                # 8 Taylor terms, no problem!

attn = SymmetryAwareTaylorApproximatedAttention(
    d_key=d_head, d_val=d_head, is_causal=True, n_taylor=n_taylor).to(DEVICE)

for tok_num in tqdm(range(n_tok)):
    q, k, v = torch.randn(3, n_head, 1, d_head, device=DEVICE)  # new query, key, value
    y = attn(q, k, v, continue_prev=tok_num > 0)                # attn over trailing toks
```

**Important**: This implementation is an initial proof of concept lacking many optimizations. Its main purpose is to validate the correctness of our mathematical derivations. Please read on before using it.



## Key Insight

The core operation of scaled dot-product attention is exponentiation of a query-key dot-product. Given query and key vectors $q, k \in \mathbb{R}^{d_K}$ and a conventional constant $c = \sqrt{d_K}$, the core operation's Taylor expansion is:

$$\begin{aligned}
\exp \left( \frac{q^\top k}{c} \right)
& = 1 + \frac{q^\top k}{c} + \frac{1}{2!} \left( \frac{q^\top k}{c} \right)^2 + \frac{1}{3!} \left( \frac{q^\top k}{c} \right)^3 + \dots \\
\\
& = \sum_{p=0}^{\infty}  \alpha_p  \left( q^\top k \right)^p, \qquad \alpha_p := \frac{1}{p! \ c^p} \\
\end{aligned}$$

Previous efforts to approximate attention via Taylor expansion have stopped at the quadratic term ($p = 2$) due to the perceived complexity of evaluating all necessary polynomial interactions for higher-degree terms.

In our paper, we show that the Taylor expansion decomposes into an expression over symmetric chains of tensor products, and their symmetric structure naturally reveals the minimal basis for all necessary polynomial interactions, enabling us to find an efficient method for computing them.

Concretely, for every $p$ in the Taylor expansion, we show that $\left( q^\top k \right)^p = \sum \left( q^{\otimes p} \right) \odot \left( k^{\otimes p} \right)$, where $\odot$ denotes elementwise (Hadamard) product, and $x^{\otimes p}$ denotes tensor (outer) product of $x$ with itself $p$ times. For example, when $p = 3$, we have $\left( q^\top k \right)^3 = \sum \left( q^{\otimes 3} \right) \odot \left( k^{\otimes 3} \right) = \sum  (q \otimes q \otimes q) \odot (k \otimes k \otimes k)$:

```python
import torch

d_key = 4                     # toy example
q, k = torch.randn(2, d_key)  # query, key

q_tensorprod_3_times = torch.einsum('i,j,k->ijk', q, q, q)  # symmetric
k_tensorprod_3_times = torch.einsum('i,j,k->ijk', k, k, k)  # symmetric

torch.allclose(
    (q @ k) ** 3,
    (q_tensorprod_3_times * k_tensorprod_3_times).sum())  # True
```

The tensors $q^{\otimes p}$ and $k^{\otimes p}$ are _symmetric_, and their elementwise product, $\left( q^{\otimes p} \right) \odot \left( k^{\otimes p} \right) = \left( q \odot k \right)^{\otimes p}$, is also _symmetric_. As our paper explains, the upper hyper-triangular region of each of these symmetric tensors contains its unique elements (analogous to a symmetric matrix's upper triangular region).

By construction, $q^{\otimes p}$ and $k^{\otimes p}$ consist of all possible degree $p$ monomials combining elements of $q$ and $k$, respectively. Therefore, the upper hyper-triangular regions of these two symmetric tensors contain the unique monomials combining elements of $q$ and $k$, respectively, that make up the _minimal basis_ for computing $(q^\top k)^p$. _All monomials outside each upper hyper-triangular region are permutations of a monomial in the region._

The upper hyper-triangular region of an order $p$ symmetric tensor is indexed by $i_1 \le i_2 \le \dots \le i_p$, and consists of $m_p = \binom{d_K + p - 1}{p}$ elements, significantly fewer than ${d_K}^p$ in the full symmetric tensor. The following figure shows symmetric tensors of increasing order. The blue region in the rightmost plot is the upper hyper-triangular region, containing the minimal monomial basis for representing $\left( q^\top k \right)^p$:

![Symmetric tensors illustration](images/symmetric_tensors.png)


Our key contribution is a maximally succinct, computationally efficient, and embarrassingly parallel feed-forward transformation, shown as `Phi()` below, implementing a feature map $\Phi_p: \mathbb{R}^{d_K} \to \mathbb{R}^{m_p}$, that takes a query or key as input and returns the $m_p$ monomials in the upper hyper-triangular region of the associated symmetric tensor, _i.e._, the minimal basis, _tightly packed in a vector_. We can then weight each basis monomial by a coefficient, equal to the corresponding number of possible permutations:

```python
from itertools import combinations_with_replacement
from sata_attention import _calculate_n_idx_permutations

p = 3

# Constants (precomputed only once, in advance):
M = torch.tensor([*combinations_with_replacement(range(d_key), p)], dtype=torch.long)
C = _calculate_n_idx_permutations(M, d_key)

# Proposed feed-forward transformation:
def Phi(x): return x[..., M].prod(dim=-1)

torch.allclose((q @ k) ** p, (Phi(q) * Phi(k) * C).sum())  # True
```

The rows of matrix $M_p \in \\{ 1, 2, \dots, d_K \\}^{m_p \times p}$ (`M`, above) contain the indices to the upper hyper-triangular region of an order $p$ symmetric tensor, sorted in ascending order. Each coefficient in $C_p$ (`C`, above) scales the corresponding monomial. For $p = 0$, $M_0$ is an empty matrix, $C_0 = 1$, and $\Phi_0(\cdot) = 1$, which we handle as a special case in PyTorch. The space and time savings grow rapidly as we increase $p$.

If we incorporate the coefficients into the feature map, we can express $\left( q^\top k \right)^p$ as a dot-product with PyTorch:

```python
sqrt_C = C.float().sqrt()
def scaledPhi(x): return Phi(x) * sqrt_C
torch.allclose((q @ k) ** p, scaledPhi(q) @ scaledPhi(k))  # True
```

In our paper, we show how to apply `Phi()` to approximate scaled dot-product attention, incurring constant cost per token, achieving orders-of-magnitude reductions in memory use and computation compared to the conventional formulation of attention. Notably, space and time complexity become inversely proportional to head size, making it cheaper to apply attention over a larger number of smaller heads.

This repository contains an implementation, along with code for verifying its correctness.


## Proof of Concept

**The implementation in this repository is an initial proof of concept, not meant for use in production.** Unlike implementations of the conventional formulation of attention, which has now benefited from nearly a decade of performance optimization work by a large community of AI researchers and practitioners, our formulation is new and has yet to benefit from comparable long-term work.

Targets for performance optimization include:

**Unnecessary Temporary Copying of Query and Key Elements**: Currently, we make $m_p \times p$ temporary copies of elements from each query and key vector, instead of pointing to those elements, for mapping the vector to $m_p$ monomial features. This is because the Python expression `x[..., M]` returns *copies* of `x`'s elements, _not views_. (Recall that matrix `M`, properly $M_p$, consists of $m_p \times p$ indices.) Copying all those elements increases temporary memory use, and can also saturate memory bandwidth, holding back performance. In principle, copying all that data is unnecessary.

**Absence of Optimizations that Exploit Hierarchical Structure of Symmetric Indices**: Currently, we do not exploit the hierarchical structure of the indices stored in matrix $M_p$. It's not hard to show that for each $p > 0$, each row of $M_{p-1}$ is a partial row of one or more rows of $M_p$, which opens additional opportunities for improving computational efficiency. In principle, it should be possible to exploit the hierarchical structure of the index matrices to reduce memory and compute use.

**Sequential Instead of Parallel Evaluation of Taylor Terms**: Currently, we evaluate Taylor Terms sequentially, instead of in parallel. We queue them on a single stream on an Nvidia GPU. As we discuss in our paper, Taylor terms can be evaluated in parallel, because they are independent of each other.

**Absence of Common On-Device Optimizations**: Currently, our focus is on validating the correctness of our formulation, not on developing a high-performance implementation. A more efficient implementation requires writing low-level on-device code (_e.g._, a fused CUDA kernel for Nvidia GPUs) that carefully handles data, both to avoid unnecessarily making temporary copies of it, and to ensure it is more frequently on faster-access memory (_e.g._, HBM instead of SRAM on Nvidia GPUs) as needed for computation.

**Absence of Additional Performance Optimizations**: Currently, we have not explored any additional performance optimizations. They include the possibility of reducing the dimensionality of higher-order feature spaces (say, for order greater than four) by applying conventional techniques, such as dropping basis components with low-magnitude coefficients, finding best-fit lower-rank basis approximations, and obtaining fast approximations of the basis via random sampling or random projections. Also, many methods for improving the space and time efficiency of the conventional formulation of attention can benefit our formulation too (_e.g._, data and parameter reuse schemes).


## Next Steps

Several directions for future work follow naturally. An important next step is to train Transformer models,
end-to-end and via transfer learning, applying our formulation of attention, and evaluate downstream
performance and convergence behavior. Ideally, such evaluations should include _new model configurations that compute attention over more heads than previously feasible_, by reducing the size of each head, because our formulation is most favorable to smaller heads. **Performing such evaluations at any meaningful scale will require _development of fused hardware kernels that target the performance issues we mention above_.**

Further opportunities include compressing higher-order feature spaces and extending our approach to other analytic kernels beyond the exponential. More generally, our work enables exploration of new approaches to address the rising demand for storage, compute, and energy, driven by growing adoption of AI services.


## Installation

Clone the repository, or download a single file: `sata_attention.py`.

The only dependency is a recent version of PyTorch.


## Toy Example

```python
import torch
from sata_attention import SymmetryAwareTaylorApproximatedAttention

DEVICE = 'cuda'  # change as needed

n_tok, d_key, d_val = (3, 64, 64)

attn = SymmetryAwareTaylorApproximatedAttention(d_key, d_val, is_causal=True, n_taylor=4)
attn = attn.to(DEVICE)

Q = torch.randn(n_tok, d_key, device=DEVICE)
K = torch.randn(n_tok, d_key, device=DEVICE)
V = torch.randn(n_tok, d_val, device=DEVICE)

Y = attn(Q, K, V)
```

Given a new token, you can compute attention at constant cost per token with:

```python
new_Q = torch.randn(1, d_key, device=DEVICE)
new_K = torch.randn(1, d_key, device=DEVICE)
new_V = torch.randn(1, d_val, device=DEVICE)

Y = attn(new_Q, new_K, new_V, continue_prev=True)  # constant cost per token
```


## Replicating Our Results

We validate the correctness of our formulation by applying our proof-of-concept implementation to sequences with up to 100M tokens. To replicate our results, install the dependencies listed in `replication_deps.txt` (e.g., `pip install -r replication_deps.txt`), and run the following from the command line:

```
python replicate_results.py
```

Note: Tested only on Linux; requires a GPU with at least 40GB of memory.


## Citing

```
@article{
heinsenkozachkov2026attention,
title={Self-Attention at Constant Cost per Token via Symmetry-Aware Taylor Approximation},
author={Franz A. Heinsen and Leo Kozachkov},
year={2026},
}
```


## Notes

We hope others find our work and our code useful.
