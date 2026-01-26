# sata_attention

Reference implementation of the *symmetry-aware Taylor approximation of attention* proposed in "Self-Attention at Constant Cost per Token via Symmetry-Aware Taylor Expansion" (Heinsen and Kozachkov, 2026). We derive our formulation of attention by decomposing the conventional formulation's Taylor expansion into expressions over symmetric chains of tensor products, and exploit their symmetry to obtain feed-forward transformations that efficiently map queries and keys to coordinates in a minimal polynomial-kernel feature basis.

# Requirements
