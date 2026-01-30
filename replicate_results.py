import math
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark

from tqdm import tqdm
from sata_attention import SymmetryAwareTaylorApproximatedAttention


# Setup

DEVICE = 'cuda'  # must be a cuda device
torch.fx.experimental._config.use_duck_shape = False  # prevents recompilations as sizes change
torch._dynamo.config.cache_size_limit = 32  # caches more recompilations to handle greater data variation

SACRIFICE_PRECISION_FOR_FASTER_EXECUTION = False  # note: does not impact *relative* run time or mem use in our tests
torch.backends.cuda.matmul.allow_tf32 = SACRIFICE_PRECISION_FOR_FASTER_EXECUTION
torch.backends.cudnn.allow_tf32 = SACRIFICE_PRECISION_FOR_FASTER_EXECUTION
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = SACRIFICE_PRECISION_FOR_FASTER_EXECUTION
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = SACRIFICE_PRECISION_FOR_FASTER_EXECUTION
torch.backends.cuda.matmul.allow_fp16_accumulation = SACRIFICE_PRECISION_FOR_FASTER_EXECUTION

ESTIMATE_PARALLEL_RUN_TIME_FOR_TPTT_LAYERS = False  # simulates parallel execution of TPTT layers

N_TIMEIT_RUNS = 7
BENCHMARK_DATA_PATH = 'benchmark_data.pt'

DEFAULT_N_TAYLOR = 4
DEFAULT_DATA_DTYPE = torch.float16

N_TAYLOR_LIST = range(1, 8 + 1)                        # number of Taylor terms to test
P_LIST = [n_taylor - 1 for n_taylor in N_TAYLOR_LIST]  # power (degree) of each Taylor term
D_HEAD_LIST = [8, 16, 32, 64]                          # head sizes to try
N_TOK_LIST = sorted(set(
    (math.floor(10 ** i) // 64 + 1) * 64
    for i in np.linspace(3, 8, 100) ))                 # approx 10**3 to 10**8, in multiples of 64

N_TOK_FOR_MEASURING_RECONSTRUCTION_ERROR = 100 * 1024  # will measure reconstruction error for every token (slow)
N_TAYLOR_LIST_FOR_RECONSTRUCTION_ERROR = [3, 4, 5, 6]  # number of Taylor terms to test for reconstruction
CHUNK_SZ_FACTOR_FOR_RECONSTRUCTION_ERROR = 2           # integer >= 1, multiplies memory footprint of reconstructions

FIG_DPI = 300
plt.rcParams['mathtext.fontset'] = 'cm'


# Bare-bones implementation of conventional attention. Feel free to replace with your own!

class BarebonesConventionalAttention(torch.nn.Module):
    "Minimal implementation of conventional scaled dot-product attention."

    def __init__(self, is_causal):
        super().__init__()
        self.is_causal = is_causal

    def forward(self, Q, K_cache, V_cache):
        A = Q @ K_cache.transpose(-2, -1) / (Q.size(-1) ** 0.5)
        if self.is_causal:
            n_qry, n_tok = (Q.size(-2), K_cache.size(-2))
            assert n_tok >= n_qry, "Not enough tokens in context for causal attention."
            mask = torch.ones(n_qry, n_qry, dtype=torch.bool, device=A.device).tril()  # tril mask for n_qry tokens
            mask = F.pad(mask, (n_tok - n_qry, 0), value=True)                         # older tokens are not masked
            A.masked_fill_(mask.logical_not(), float("-inf"))
        return A.softmax(dim=-1) @ V_cache


# Code for running benchmarks

def benchmark_hid_state_sz_vs_conventional():
    print('Benchmarking hidden state size vs. conventional.')
    print('Evaluating sequences of up to 10**{:.0f} tokens.'.format(math.log10(max(N_TOK_LIST))))
    runs = []
    for d_head in D_HEAD_LIST:
        new_attn = SymmetryAwareTaylorApproximatedAttention(d_head, d_head, is_causal=False).eval()
        for n_tok in tqdm(N_TOK_LIST, desc=f'd_head={d_head}'):
           runs.append({
                'd_head': d_head,
                'n_tok': n_tok,
                'old_hid_state_sz': n_tok * d_head * 2,  # size of key-value cache
                'new_hid_state_sz': new_attn.get_hidden_state_sizes()['Total'],
            })
    return runs


def benchmark_flops_per_tok_vs_conventional():
    print('---\nBenchmarking FLOPs per token vs. conventional attention.')
    print('Evaluating sequences of up to 10**{:.0f} tokens.'.format(math.log10(max(N_TOK_LIST))))

    # We measure FLOPs *per token*, not per sequence.
    # We measure FLOPs only for Attention(Q, K V), not for anything else.
    # We assume queries, keys, and values have already been computed.

    def _old_attn_forward_flops(n_qry, n_tok, d_head, n_heads):
        # For conventional attention, we measure FLOPs w/method from DeepMind's Scaling Laws/Chinchilla paper
        # (https://arxiv.org/abs/2203.15556). See also https://www.adamcasson.com/posts/transformer-flops
        flops_qk_logits = 2 * n_qry * n_tok * d_head * n_heads
        flops_softmax = 2 * n_qry * n_tok * d_head * n_heads
        flops_reduction = 2 * n_qry * n_tok * d_head * n_heads
        return flops_qk_logits + flops_softmax + flops_reduction
    
    runs = []
    for d_head in D_HEAD_LIST:
        new_attn = SymmetryAwareTaylorApproximatedAttention(d_head, d_head, is_causal=False).to(device=DEVICE).eval()
        for n_tok in tqdm(N_TOK_LIST, desc=f'd_head={d_head}'):
           runs.append({
                'd_head': d_head,
                'n_tok': n_tok,
                'old_FLOPs_per_query_head': _old_attn_forward_flops(n_qry=1, n_tok=n_tok, d_head=d_head, n_heads=1),
                'new_FLOPs_per_query_head': new_attn.get_forward_FLOPs_per_query_head()['Total'],
            })
    
    torch.cuda.empty_cache()
    return runs


def benchmark_hid_state_sz_and_flops_per_tok_for_multiple_heads():
    print('---\nBenchmarking hidden state size and FLOPs per token for multi-head configurations.')
    print('Evaluating embedding sizes up to 8192.')
    runs = []
    for d_head in np.arange(8, max(D_HEAD_LIST) + 1, 8):
        new_attn = SymmetryAwareTaylorApproximatedAttention(d_head, d_head, is_causal=False, n_taylor=DEFAULT_N_TAYLOR).eval()
        _hid_sz_one_head = new_attn.get_hidden_state_sizes()['Total']
        _flops_one_head = new_attn.get_forward_FLOPs_per_query_head()['Total']
        for d_emb in tqdm(np.arange(1, 1 + 8192 // 1024) * 1024, desc=f'd_head={d_head}'):
            n_heads = d_emb // d_head
            runs.append({
                'd_emb': int(d_emb),
                'n_heads': int(n_heads),
                'd_head': int(d_head),
                'hid_sz_per_tok_all_heads': int(_hid_sz_one_head * n_heads),
                'FLOPs_per_tok_all_heads': int(_flops_one_head * n_heads),
            })
    return runs


def benchmark_mem_use_per_tok_vs_conventional():
    print('---\nBenchmarking memory use per token vs. conventional attention.')
    print('Applying attention {} times to {} sample sequences with up to 10**{:.0f} tokens.'.format(
        N_TIMEIT_RUNS, len(N_TOK_LIST), math.log10(max(N_TOK_LIST))))
    print('Lazily compiling all modules.')

    # Cache more recompilations to handle more autocasting variations as sequences get longer:
    _saved_dynamo_cache_size_limit = torch._dynamo.config.cache_size_limit
    torch._dynamo.config.cache_size_limit = 32

    # Instantiate conventional attention:
    old_attn = BarebonesConventionalAttention(is_causal=False)
    old_attn = torch.compile(old_attn, dynamic=True)
    old_attn.eval()

    runs = []
    for d_head in D_HEAD_LIST:

        # Instantiate reformulated attention:
        new_attn = SymmetryAwareTaylorApproximatedAttention(d_head, d_head, is_causal=False, n_taylor=DEFAULT_N_TAYLOR)
        new_attn = torch.compile(new_attn, dynamic=True)
        new_attn.to(device=DEVICE).eval()

        # Create fixed-size hidden states by passing one query, key, and value:
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=True):
            _ = new_attn(*torch.randn(3, 1, d_head, device=DEVICE))

        for n_tok in tqdm(N_TOK_LIST, desc=f'd_head={d_head}'):
    
            torch.cuda.empty_cache()
            torch.cuda.memory.reset_peak_memory_stats(device=DEVICE)
            Q = torch.randn(1, d_head, dtype=DEFAULT_DATA_DTYPE, device=DEVICE)
            K_cache = torch.randn(n_tok, d_head, dtype=DEFAULT_DATA_DTYPE, device=DEVICE)
            V_cache = torch.randn(n_tok, d_head, dtype=DEFAULT_DATA_DTYPE, device=DEVICE)
            with torch.no_grad(), torch.autocast(device_type='cuda', enabled=True):
                Y = old_attn(Q, K_cache, V_cache)  # one query given KV cache
            old_max_mem = torch.cuda.memory.max_memory_allocated(device=DEVICE)
            del Q, K_cache, V_cache, Y
    
            torch.cuda.empty_cache()
            torch.cuda.memory.reset_peak_memory_stats(device=DEVICE)
            Q = torch.randn(1, d_head, dtype=DEFAULT_DATA_DTYPE, device=DEVICE)
            K = torch.randn(1, d_head, dtype=DEFAULT_DATA_DTYPE, device=DEVICE)
            V = torch.randn(1, d_head, dtype=DEFAULT_DATA_DTYPE, device=DEVICE)
            with torch.no_grad(), torch.autocast(device_type='cuda', enabled=True):
                Y = new_attn(Q, K, V, continue_prev=True)  # use prev hid states
            new_max_mem = torch.cuda.memory.max_memory_allocated(device=DEVICE)
            del Q, K, V, Y
    
            runs.append({
                'd_head': d_head,
                'n_tok': n_tok,
                'old_max_mem': old_max_mem,
                'new_max_mem': new_max_mem,
            })

        del new_attn
        torch.cuda.empty_cache()

    return runs


def benchmark_run_time_per_tok_vs_conventional():
    print('---\nBenchmarking run-time per token vs. conventional attention.')
    print('Applying attention {} times to {} sample sequences with up to 10**{:.0f} tokens.'.format(
        N_TIMEIT_RUNS, len(N_TOK_LIST), math.log10(max(N_TOK_LIST))))
    print('Lazily compiling all modules.')

    # Instantiate conventional attention:
    old_attn = BarebonesConventionalAttention(is_causal=False)
    old_attn = torch.compile(old_attn, dynamic=True)
    old_attn.eval()

    runs = []
    for d_head in D_HEAD_LIST:

        # Instantiate reformulated attention:
        new_attn = SymmetryAwareTaylorApproximatedAttention(d_head, d_head, is_causal=False, n_taylor=DEFAULT_N_TAYLOR)
        if ESTIMATE_PARALLEL_RUN_TIME_FOR_TPTT_LAYERS:
            new_attn.tptts = new_attn.tptts[-1:]  # will execute only the slowest (highest-order) tptt layer
        new_attn = torch.compile(new_attn, dynamic=True)
        new_attn.to(device=DEVICE).eval()

        # Create fixed-size hidden states by passing one query, key, and value:
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=True):
            _ = new_attn(*torch.randn(3, 1, d_head, device=DEVICE))

        for n_tok in tqdm(N_TOK_LIST, desc=f'd_head={d_head}'):
            torch.cuda.empty_cache()
            Q = torch.randn(1, d_head, dtype=DEFAULT_DATA_DTYPE, device=DEVICE)
            K_cache = torch.randn(n_tok, d_head, dtype=DEFAULT_DATA_DTYPE, device=DEVICE)
            V_cache = torch.randn(n_tok, d_head, dtype=DEFAULT_DATA_DTYPE, device=DEVICE)
            K = K_cache[..., -1:, :].clone()
            V = V_cache[..., -1:, :].clone()
            runs.append({
                'd_head': d_head,
                'n_tok': n_tok,
                'old_time': torch.utils.benchmark.Timer(
                    stmt="with torch.no_grad(), torch.autocast(device_type='cuda', enabled=True): Y = old_attn(Q, K_cache, V_cache)",  # one query given KV cache
                    globals={'Q': Q, 'K_cache': K_cache, 'V_cache': V_cache, 'old_attn': old_attn, 'torch': torch, },
                ).timeit(N_TIMEIT_RUNS).mean,
                'new_time': torch.utils.benchmark.Timer(
                    stmt="with torch.no_grad(), torch.autocast(device_type='cuda', enabled=True): Y = new_attn(Q, K, V, continue_prev=True)",  # use prev hid state
                    globals={'Q': Q, 'K': K, 'V': V, 'new_attn': new_attn, 'torch': torch, },
                ).timeit(N_TIMEIT_RUNS).mean,

            })
            del Q, K, V, K_cache, V_cache

        del new_attn
        torch.cuda.empty_cache()

    del old_attn
    torch.cuda.empty_cache()

    return runs


def benchmark_reconstruction_error_vs_conventional():
    n_tok = N_TOK_FOR_MEASURING_RECONSTRUCTION_ERROR  # for brevity's sake
    print('---\nBenchmarking reconstruction error vs. conventional self-attention at Float64 precision.')
    print(f'Applying self-attention to sequences of {n_tok // 1024}K tokens (in chunks to limit memory use).')

    # Instantiate conventional attention:
    old_attn = BarebonesConventionalAttention(is_causal=True)
    old_attn.eval()

    max_d_head = max(D_HEAD_LIST)
    runs = []
    for d_head in D_HEAD_LIST:
        torch.cuda.empty_cache()
        n_heads = max_d_head // d_head  # number of heads (processed in parallel)
        chunk_sz = CHUNK_SZ_FACTOR_FOR_RECONSTRUCTION_ERROR * max_d_head // d_head  # number of tokens per chunk (processed in parallel)
        Q = torch.randn(n_heads, n_tok, d_head, dtype=DEFAULT_DATA_DTYPE, device=DEVICE)
        K = torch.randn(n_heads, n_tok, d_head, dtype=DEFAULT_DATA_DTYPE, device=DEVICE)
        V = torch.randn(n_heads, n_tok, d_head, dtype=DEFAULT_DATA_DTYPE, device=DEVICE)
        for n_taylor in N_TAYLOR_LIST_FOR_RECONSTRUCTION_ERROR:

            # Instantiate reformulated attention:
            new_attn = SymmetryAwareTaylorApproximatedAttention(d_head, d_head, is_causal=True, n_taylor=n_taylor)
            new_attn.to(device=DEVICE).eval()

            tgt_Y, new_Y = ([], [])
            for t in tqdm(range(0, n_tok, chunk_sz), desc=f'd_head={d_head}, n_heads={n_heads}, chunk_sz={chunk_sz}, n_taylor={n_taylor}'):
                new_Q = Q[..., t:(t + chunk_sz), :]   # [n_heads, chunk_sz, d_head]
                new_K = K[..., t:(t + chunk_sz), :]   # [n_heads, chunk_sz, d_head]
                new_V = V[..., t:(t + chunk_sz), :]   # [n_heads, chunk_sz, d_head]
                K_cache = K[..., :(t + chunk_sz), :]  # [n_heads, t + chunk_sz, d_head], including new_K
                V_cache = V[..., :(t + chunk_sz), :]  # [n_heads, t + chunk_sz, d_head], including new_V
                with torch.no_grad():  # no autocasting for Float64 targets
                    args = (arg.to(torch.float64) for arg in (new_Q, K_cache, V_cache))
                    tgt_Y.append(old_attn(*args).to(device='cpu'))
                with torch.no_grad(), torch.autocast(device_type='cuda', enabled=True):
                    new_Y.append(new_attn(new_Q, new_K, new_V, continue_prev=t > 0).to(dtype=torch.float64).to(device='cpu'))

            runs.append({
                'd_head': d_head,
                'n_taylor': n_taylor,
                'n_heads': n_heads,
                'tgt_Y': torch.cat(tgt_Y, dim=-2),
                'new_Y': torch.cat(new_Y, dim=-2),
            })

            del new_attn, tgt_Y, new_Y
            torch.cuda.empty_cache()

        del Q, K, V
        torch.cuda.empty_cache()
    
    benchmark_data['reconstruction_error_vs_conventional'] = runs
    return runs


# Code for generating and saving figures

def generate_and_save_fig_reduction_achieved_by_tight_packing(benchmark_data):
    fig, axes = plt.subplots(ncols=2, sharex=True, figsize=(10, 4), layout='constrained')
    for d_head in D_HEAD_LIST:

        axis = axes[0]
        full_hid_state_szs = np.array([d_head ** p * (d_head + 1) for p in P_LIST])
        packed_hid_state_szs = np.array([math.comb(d_head + p - 1, p) * (d_head + 1) for p in P_LIST])
        ratios = packed_hid_state_szs / full_hid_state_szs 
        pd.Series(ratios, index=N_TAYLOR_LIST).plot(ax=axis, label=d_head, marker='o', lw=2, alpha=0.7)
        axis.set(yscale='log', title='Reduction in Hidden State Size\nAchieved by Tight Packing', ylabel='Ratio, Tightly Packed to Unpacked')
        axis.set(xlabel='Taylor Term', xticks=N_TAYLOR_LIST)
        axis.grid(axis='y')
        axis.legend(title='Head Size')
    
        axis = axes[1]
        full_flops = np.array([5 * d_head ** p * (d_head + 1) for p in P_LIST])
        packed_flops = np.array([(2 + 2 * (p + 1) + 4 * d_head) * math.comb(d_head + p - 1, p) for p in P_LIST])
        ratios = packed_flops / full_flops
        ratios[0] = 1
        pd.Series(ratios, index=N_TAYLOR_LIST).plot(ax=axis, label=d_head, marker='o', lw=2, alpha=0.7)
        axis.set(yscale='log', title='Reduction in FLOPs per Token\nAchieved by Tight Packing', ylabel='Ratio, Tightly Packed to Unpacked')
        axis.set(xlabel='Taylor Term', xticks=N_TAYLOR_LIST)
        axis.grid(axis='y')
        axis.legend(title='Head Size')

    fig.savefig('fig_reduction_achieved_by_tight_packing.png', dpi=FIG_DPI)
    plt.close(fig)


def generate_and_save_fig_scaling_constant_versus_float_resolution(benchmark_data):
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4), layout='constrained')

    axis = axes[0]
    for d_head in D_HEAD_LIST:
        alphas = [1 / (math.factorial(p) * (d_head ** (p / 2))) for p in P_LIST]
        pd.Series(alphas, index=N_TAYLOR_LIST).plot(ax=axis, label=d_head, marker='o', lw=2, alpha=0.7)
        axis.set(ylabel=r'Scalar Value', yscale='log', xlabel='Taylor Term', xticks=N_TAYLOR_LIST)
        axis.set(title=r'Scaling Constant $\alpha_p$ by Taylor Term')
        axis.grid(axis='y')
        axis.legend(title='Head Size')

    axis = axes[1]
    float_fmt_tups = [
        ('BFloat16', torch.bfloat16),
        ('Float16', torch.float16),
        ('Float32', torch.float32),
    ]
    pd.Series(
        data=[torch.finfo(tup[1]).resolution for tup in float_fmt_tups],
        index=[tup[0] for tup in float_fmt_tups],
    ).plot(ax=axis, marker='o', lw=2, alpha=0.7, color='black')
    axis.set(ylabel=r'Resolution', yscale='log', xlabel='Floating-Point Format', xticks=range(len(float_fmt_tups)), xlim=(-0.5, len(float_fmt_tups) -0.5))
    axis.set(title='Resolution of Common Floating-Point Formats')
    axis.grid(axis='y')

    axes[1].set(ylim=axes[0].get_ylim())

    fig.savefig('fig_scaling_constant_versus_float_resolution.png', dpi=FIG_DPI)
    plt.close(fig)


def generate_and_save_fig_hidden_state_size_and_flops_vs_conventional(benchmark_data):
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4), layout='constrained')

    axis = axes[0]
    df = pd.DataFrame(benchmark_data['hid_state_sz_vs_conventional'])
    df['ratio'] = df.new_hid_state_sz / df.old_hid_state_sz
    for d_head in D_HEAD_LIST:
        df[df.d_head == d_head].set_index('n_tok').ratio.plot(ax=axis, label=str(d_head), lw=2, alpha=0.7)
    axis.set(xscale='log', yscale='log', xlabel='Number of Tokens in Context', ylabel=r"Ratio, Ours to Conventional (KV Cache)")
    axis.set(title=f'Relative Size of Hidden State ({DEFAULT_N_TAYLOR} Taylor Terms)\nCompared to Conventional Formulation')
    axis.legend(title='Head Size')
    axis.grid(axis='y')

    axis = axes[1]
    df = pd.DataFrame(benchmark_data['flops_per_tok_vs_conventional'])
    df['ratio'] = df.new_FLOPs_per_query_head / df.old_FLOPs_per_query_head
    for d_head in D_HEAD_LIST:
        df[df.d_head == d_head].set_index('n_tok').ratio.plot(ax=axis, label=str(d_head), lw=2, alpha=0.7)
    axis.set(xscale='log', yscale='log', xlabel='Number of Tokens in Context', ylabel=r"Ratio, Ours to Conventional")
    axis.set(title=f'Relative FLOPs per Token ({DEFAULT_N_TAYLOR} Taylor Terms)\nCompared to Conventional Formulation')
    axis.legend(title='Head Size')
    axis.grid(axis='y')

    axes[1].set(ylim=axes[0].get_ylim())

    fig.savefig('fig_hidden_state_size_and_flops_vs_conventional.png', dpi=FIG_DPI)
    plt.close(fig)


def generate_and_save_fig_poc_benchmarks_against_conventional(benchmark_data):
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True, layout='constrained')
    
    axis = axes[0]
    df = pd.DataFrame(benchmark_data['mem_use_per_tok_vs_conventional'])
    df['ratio'] = df.new_max_mem / df.old_max_mem
    for d_head in D_HEAD_LIST:
        df[df.d_head == d_head].set_index('n_tok').ratio.plot(ax=axis, label=str(d_head), lw=2, alpha=0.7)
    axis.set(xscale='log', yscale='log', xlabel='Number of Tokens in Context', ylabel="Ratio, Our Proof-of-Concept to\nConventional Implementation")
    axis.set_title(f'Relative Peak Memory Allocated per Token\n({DEFAULT_N_TAYLOR} Taylor Terms, Forward Pass, Autocasting, Nvidia GPU)', fontsize=11)
    axis.legend(title='Head Size')
    axis.grid(axis='y')
    
    axis = axes[1]
    df = pd.DataFrame(benchmark_data['run_time_per_tok_vs_conventional'])
    df['ratio'] = df.new_time / df.old_time
    for d_head in D_HEAD_LIST:
        df[df.d_head == d_head].set_index('n_tok').ratio.plot(ax=axis, label=str(d_head), lw=2, alpha=0.7)
    axis.set(xscale='log', yscale='log', xlabel='Number of Tokens in Context', ylabel="Ratio, Our Proof-of-Concept to\nConventional Implementation")
    axis.set_title(f'Relative Run Time per Token, Mean of {N_TIMEIT_RUNS} Runs\n({DEFAULT_N_TAYLOR} Taylor Terms, Forward Pass, Autocasting, Nvidia GPU)', fontsize=11)
    axis.legend(title='Head Size')
    axis.grid(axis='y')

    # axes[1].set(ylim=tuple(axes[0].get_ylim()))

    fig.savefig('fig_poc_benchmarks_against_conventional.png', dpi=FIG_DPI)
    plt.close(fig)


def generate_and_save_fig_hidden_sz_and_flops_for_multiple_heads(benchmark_data):
    fig, axes = plt.subplots(figsize=(10, 4.5), ncols=2, layout='constrained')
    fig.suptitle(f'Hidden State Size and FLOPs per Token for Multi-Head Configurations ({DEFAULT_N_TAYLOR} Taylor Terms)\n')

    axis = axes[0]
    df = pd.DataFrame(benchmark_data['hid_state_sz_and_flops_per_tok_for_multiple_heads'])
    df = df.set_index(['d_emb', 'd_head'])['hid_sz_per_tok_all_heads'].unstack()
    _obj = axis.pcolor(df, norm='log', edgecolors='white', lw=1)
    _cbar = fig.colorbar(_obj, ax=axis, shrink=0.7, label='Number of Elements\n')  # extra new line for spacing
    axis.set(yticks=np.arange(1, 9) - 0.5, yticklabels=np.arange(1, 9) * 1024)
    axis.set(xticks=df.columns / 8 - 0.5, xticklabels=df.columns)
    axis.set(title='Combined Hidden State Size')
    axis.set(xlabel='Head Size', ylabel='Embedding Size')
    axis.set_aspect('equal')

    axis = axes[1]
    df = pd.DataFrame(benchmark_data['hid_state_sz_and_flops_per_tok_for_multiple_heads'])
    df = df.set_index(['d_emb', 'd_head'])['FLOPs_per_tok_all_heads'].unstack()
    _obj = axis.pcolor(df, norm='log', edgecolors='white', lw=1)
    _cbar = fig.colorbar(_obj, ax=axis, shrink=0.7, label='FLOPs per Token')
    axis.set(yticks=np.arange(1, 9) - 0.5, yticklabels=np.arange(1, 9) * 1024)
    axis.set(xticks=df.columns / 8 - 0.5, xticklabels=df.columns)
    axis.set(title='Combined FLOPs per Token')
    axis.set(xlabel='Head Size', ylabel='\nEmbedding Size')  # extra new line for spacing
    axis.set_aspect('equal')

    fig.savefig('fig_hidden_sz_and_flops_for_multiple_heads.png', dpi=FIG_DPI)
    plt.close(fig)


def generate_and_save_figs_reconstruction_error_vs_conventional(benchmark_data):
    n_tok = N_TOK_FOR_MEASURING_RECONSTRUCTION_ERROR  # for brevity's sake
    finite_min_err = torch.tensor(torch.finfo(torch.float64).resolution, dtype=torch.float64).log10()
    runs = benchmark_data['reconstruction_error_vs_conventional']
    d_head_to_color = { d_head: f'C{i}' for i, d_head in enumerate(D_HEAD_LIST) }

    # Histograms:
    n_cols = len(D_HEAD_LIST)
    n_rows = len(N_TAYLOR_LIST_FOR_RECONSTRUCTION_ERROR)
    fig, axes = plt.subplots(figsize=(3 * n_cols, 3 * n_rows), ncols=n_cols, nrows=n_rows, sharex=False, sharey=True, layout='constrained')
    fig.suptitle(
        'Histograms of Elementwise Error Reconstructing Conventional Self-Attention at Float64 Precision over {}K Tokens\n'.format(n_tok // 1024)
            + '(Decimal Orders of Magnitude, Proof-of-Concept Implementation, Autocasting, Nvidia GPU)\n',
        fontsize=14
    )
    for axis, run in zip(axes.T.flatten(), runs):
        axis.set(title='Head Size: {}, Taylor Terms: {}'.format(run['d_head'], run['n_taylor']))
        errs = (run['tgt_Y'] - run['new_Y']).abs().log10().flatten().maximum(finite_min_err)
        color = d_head_to_color[run['d_head']]
        pd.Series(errs).hist(bins=100, ax=axis, cumulative=False, density=True, alpha=0.3, grid=False, color=color)
        axis.vlines(errs.quantile(torch.tensor(0.5).to(errs.dtype)), 0, 1.5, alpha=0.7, color=color, label='median')
        axis.set(xlim=(-6.25, 0.25), xticks=range(-6, 1), ylim=(0, 1.1), yticks=[0, 0.25, 0.5, 0.75, 1])
        axis.grid(axis='y')
        axis.set_xlabel(r'$\log_{10} | Y - \hat{Y} |$', fontsize=12)
        axis.legend()

    for row_num in range(n_rows):
        axes[row_num, 0].set(ylabel='Density')   

    fig.savefig('fig_reconstruction_error_vs_conventional.png', dpi=FIG_DPI)
    plt.close(fig)

    # Errors by token position:
    n_cols = len(N_TAYLOR_LIST_FOR_RECONSTRUCTION_ERROR)
    n_rows = len(D_HEAD_LIST)
    fig, axes = plt.subplots(figsize=(3 * n_cols, 3 * n_rows), ncols=n_cols, nrows=n_rows, sharex=False, sharey=True, layout='constrained')
    fig.suptitle(
        'Elementwise Error Reconstructing Conventional Self-Attention at Float64 Precision by Context Length\n'
            + '(Decimal Orders of Magnitude, Proof-of-Concept Implementation, Autocasting, Nvidia GPU)\n',
        fontsize=14
    )
    for axis, run in zip(axes.flatten(), runs):
        axis.set(title='Head Size: {}, Taylor Terms: {}'.format(run['d_head'], run['n_taylor']))
        errs = (run['tgt_Y'] - run['new_Y']).abs().log10().maximum(finite_min_err)  # [n_heads, n_tok, d_head]
        errs = errs.movedim(-2, 0)  # [n_tok, n_heads, d_head]
        chunk_sz = 64  # will group tokens in chunks of this size for computing statistics
        n_chunks = n_tok // chunk_sz
        errs = errs.reshape(n_chunks, -1)  # [n_chunks, (chunk_sz * n_heads * d_head)]
        color = d_head_to_color[run['d_head']]
        plot_obj = axis.plot(errs.quantile(0.5, dim=-1), alpha=0.7, color=color, label='median')
        axis.fill_between(range(n_chunks), errs.quantile(0.05, dim=-1), errs.quantile(0.95, dim=-1), lw=0, alpha=0.3, color=color, label='5% to 95%')
        xticks = [*range(0, 1 + n_chunks, n_chunks // 4)]
        axis.set(xlabel='Tokens in Context', xticks=xticks)
        axis.set(xticklabels=[('0' if t == 0 else f'{t * chunk_sz // 1024}K') for t in xticks])
        axis.set(ylim=(-6.25, 0.25), yticks=range(-6, 1))
        axis.grid(axis='y')
        axis.legend()

    for row_num in range(n_rows):
        axes[row_num, 0].set_ylabel(r'$\log_{10} | Y - \hat{Y} |$', fontsize=12)

    fig.savefig('fig_reconstruction_error_by_token_position_vs_conventional.png', dpi=FIG_DPI)
    plt.close(fig)


# Code for calling all functions

if __name__ == '__main__':

    # Load benchmark data, if previously saved:
    if pathlib.Path(BENCHMARK_DATA_PATH).is_file():
        print(f'Loading saved benchmark data from ./{BENCHMARK_DATA_PATH}.')
        benchmark_data = torch.load(BENCHMARK_DATA_PATH)
    else:
        benchmark_data = {}

    # Run benchmarks, replacing any previously saved benchmarks (comment out benchmarks to skip):
    print('---\nRunning benchmarks.')
    benchmark_data['reconstruction_error_vs_conventional'] = benchmark_reconstruction_error_vs_conventional()
    benchmark_data['run_time_per_tok_vs_conventional'] = benchmark_run_time_per_tok_vs_conventional()
    benchmark_data['mem_use_per_tok_vs_conventional'] = benchmark_mem_use_per_tok_vs_conventional()
    benchmark_data['hid_state_sz_vs_conventional'] = benchmark_hid_state_sz_vs_conventional()
    benchmark_data['flops_per_tok_vs_conventional'] = benchmark_flops_per_tok_vs_conventional()
    benchmark_data['hid_state_sz_and_flops_per_tok_for_multiple_heads'] = benchmark_hid_state_sz_and_flops_per_tok_for_multiple_heads()

    # Save updated benchmark data:
    print(f"---\nSaving benchmark data to ./{BENCHMARK_DATA_PATH}. Load it with torch.load('{BENCHMARK_DATA_PATH}').")
    torch.save(benchmark_data, BENCHMARK_DATA_PATH)

    # Generate and save figures:
    print(f'---\nGenerating and saving figures to ./{BENCHMARK_DATA_PATH}/fig_*.')
    generate_and_save_figs_reconstruction_error_vs_conventional(benchmark_data)
    generate_and_save_fig_poc_benchmarks_against_conventional(benchmark_data)
    generate_and_save_fig_hidden_sz_and_flops_for_multiple_heads(benchmark_data)
    generate_and_save_fig_reduction_achieved_by_tight_packing(benchmark_data)
    generate_and_save_fig_scaling_constant_versus_float_resolution(benchmark_data)
    generate_and_save_fig_hidden_state_size_and_flops_vs_conventional(benchmark_data)