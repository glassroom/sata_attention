# sata_attention

Reference implementation of symmetry-aware Taylor approximation of attention, as proposed in "Self-Attention at Constant Cost per Token via Symmetry-Aware Taylor Expansion" (Heinsen and Kozachkov, 2026). We have decomposed the Taylor expansion of scaled-dot-product attention into expressions over symmetric chains of tensor products, and have exploited their symmetry to obtain feed-forward transformations that efficiently map queries and keys to coordinates in a minimal polynomial-kernel feature basis.


## Instalation

1. Clone this repository.

2. Start a Python virual environment.

3. Install the dependencies listed in `requirements.txt` (e.g., `pip install -r requirements.txt`).


## Toy Example

```python
import torch
from sata_attention import ReformulatedAttention

DEVICE = 'cuda'  # change as needed

n_tok, d_key, d_val = (5, 4, 4)

attn = ReformulatedAttention(d_key, d_val, is_causal=True, n_taylor=8).cuda(DEVICE)

Q = torch.randn(n_tok, d_key, device=DEVICE)
K = torch.randn(n_tok, d_key, device=DEVICE)
V = torch.randn(n_tok, d_val, device=DEVICE)

Y = attn(Q, K, V)
```


## Replicate Our Results

Run the following from the command line (tested only on Linux):

```
python replicate_results.py
```


## Citing

Citation info goes here
