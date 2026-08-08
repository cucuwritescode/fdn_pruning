"""
Experiment 21: Orthogonal FDN Feedback Family Benchmark
========================================================
Reproducible benchmarking harness for orthogonal feedback matrix families
in the tiny-FDN regime.

Families:
  1. Dense orthogonal (QR baseline)
  2. CMV / real unitary-Hessenberg
  3. K-Householder chain (K=1,2,3,4)
  4. Constant-depth staged orthogonal (2x2 and 4x4 blocks, L=2..6)

Metrics per matrix:
  - Orthogonality error
  - Explicit density / nnz
  - Eigenvalues / eigenangles
  - Modal std (dB)
  - Graph diameter & avg shortest path
  - IR generation (lossless + attenuated)
  - Echo density growth
  - Runtime benchmarking (factorised vs dense apply)

Author: Facundo Franchino
"""

import numpy as np
import json
import csv
import os
import time
import warnings
from pathlib import Path
from scipy.linalg import block_diag
from scipy.signal import find_peaks
from scipy.sparse.csgraph import shortest_path

#============================================================================
#reproducibility
#============================================================================

MASTER_SEED = 42
FS = 48000
IR_DURATION = 0.25       #seconds for ir renders (short for speed; increase for listening)
AUDIO_DURATION = 1.0     #seconds for audio renders
TIMING_REPEATS = 500     #repeats for apply benchmarking
TIMING_WARMUP = 50       #warmup iterations

#delay sets, identical across families for fair comparison
DELAY_SETS = {
    4:  [97, 113, 127, 139],
    6:  [97, 113, 127, 139, 151, 163],
    8:  [97, 113, 127, 139, 151, 163, 173, 191],
    12: [97, 113, 127, 139, 151, 163, 173, 191, 199, 211, 223, 233],
    16: [97, 113, 127, 139, 151, 163, 173, 191, 199, 211, 223, 233, 241, 251, 263, 269],
    32: [int(p) for p in [97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,
                          179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263]],
    64: [int(p) for p in [97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,
                          179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,
                          269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,
                          367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457]],
}

#fdn conditions
T60_SECONDS = 2.0
ALPHA_LOWPASS = 0.2   #one-pole absorption coefficient

#sizes to benchmark
SIZES_PRIMARY = [4, 6, 8, 12, 16]
SIZES_OPTIONAL = [32, 64]

#============================================================================
#i. family implementations
#============================================================================

#--- dense orthogonal (qr baseline) ---

def gen_dense_orthogonal(N, rng):
    """Haar-distributed random orthogonal matrix via QR."""
    H = rng.standard_normal((N, N))
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q

def apply_dense(A, x):
    """Dense matrix-vector multiply."""
    return A @ x

def family_dense_orthogonal(N, seed):
    """Generate a dense orthogonal instance."""
    rng = np.random.default_rng(seed)
    A = gen_dense_orthogonal(N, rng)
    params = {"seed": seed}
    return {
        "family": "dense_orthogonal",
        "N": N,
        "matrix": A,
        "params": params,
        "apply_fn": lambda x, _A=A: apply_dense(_A, x),
        "apply_name": "dense_matvec",
    }


#--- cmv / real unitary-hessenberg ---

def build_cmv(thetas, N):
    """
    CMV matrix from N-1 Schur parameters.
    Two parallel Givens layers (even + odd pairs).
    A = Layer_even @ Layer_odd
    """
    assert len(thetas) == N - 1
    L_even = np.eye(N)
    for k in range(0, N - 1, 2):
        c, s = np.cos(thetas[k]), np.sin(thetas[k])
        L_even[k, k] = c;     L_even[k, k+1] = -s
        L_even[k+1, k] = s;   L_even[k+1, k+1] = c
    L_odd = np.eye(N)
    for k in range(1, N - 1, 2):
        c, s = np.cos(thetas[k]), np.sin(thetas[k])
        L_odd[k, k] = c;     L_odd[k, k+1] = -s
        L_odd[k+1, k] = s;   L_odd[k+1, k+1] = c
    return L_even @ L_odd

def apply_cmv(thetas, x, N):
    """
    O(N) factorised CMV apply: odd layer then even layer.
    """
    y = x.copy()
    #odd layer first
    for k in range(1, N - 1, 2):
        c, s = np.cos(thetas[k]), np.sin(thetas[k])
        yk, yk1 = y[k], y[k+1]
        y[k]   = c * yk - s * yk1
        y[k+1] = s * yk + c * yk1
    #even layer
    for k in range(0, N - 1, 2):
        c, s = np.cos(thetas[k]), np.sin(thetas[k])
        yk, yk1 = y[k], y[k+1]
        y[k]   = c * yk - s * yk1
        y[k+1] = s * yk + c * yk1
    return y

def family_cmv(N, seed):
    """Generate a CMV instance with random angles."""
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0, 2 * np.pi, N - 1)
    A = build_cmv(thetas, N)
    params = {"seed": seed, "thetas": thetas.tolist()}
    return {
        "family": "cmv",
        "N": N,
        "matrix": A,
        "params": params,
        "apply_fn": lambda x, _t=thetas, _N=N: apply_cmv(_t, x, _N),
        "apply_name": "cmv_factorised",
    }


#--- k-householder chain ---

def apply_k_householder(x, vectors):
    """Apply product of K Householder reflections: A = H_K ... H_1."""
    y = x.copy()
    for v in vectors:
        dot = np.dot(v, y)
        y = y - 2.0 * dot * v
    return y

def build_dense_from_k_householder(vectors):
    """Explicitly form N x N matrix from K unit vectors."""
    K, N = vectors.shape
    A = np.eye(N)
    for v in vectors:
        H = np.eye(N) - 2.0 * np.outer(v, v)
        A = H @ A
    return A

def family_k_householder(N, K, seed):
    """Generate K-Householder instance with random unit vectors."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((K, N))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    A = build_dense_from_k_householder(vectors)
    params = {"seed": seed, "K": K, "vectors": vectors.tolist()}
    return {
        "family": f"householder_K{K}",
        "N": N,
        "matrix": A,
        "params": params,
        "apply_fn": lambda x, _v=vectors: apply_k_householder(x, _v),
        "apply_name": f"householder_K{K}_factorised",
    }


#--- constant-depth staged orthogonal ---

def _perfect_shuffle_perm(N):
    """Perfect shuffle: interleave first and second halves."""
    half = N // 2
    perm = np.zeros(N, dtype=int)
    for i in range(half):
        perm[2*i] = i
        perm[2*i+1] = half + i
    if N % 2 == 1:
        perm[N-1] = N-1
    return perm

def _bit_reversal_perm(N):
    """Bit-reversal permutation (works best for powers of 2, wraps for others)."""
    bits = max(1, int(np.ceil(np.log2(N)))) if N > 1 else 1
    perm = np.zeros(N, dtype=int)
    for i in range(N):
        rev = int(format(i, f'0{bits}b')[::-1], 2)
        perm[i] = rev % N
    #fix collisions to ensure a valid permutation
    seen = set()
    out = []
    for p in perm:
        while p in seen:
            p = (p + 1) % N
        seen.add(p)
        out.append(p)
    return np.array(out, dtype=int)

def _stride_perm(N, stride=2):
    """Stride-based permutation: i -> (i * stride) % N, with fixup."""
    perm = np.array([(i * stride) % N for i in range(N)], dtype=int)
    #ensure valid permutation
    seen = set()
    out = []
    for p in perm:
        while p in seen:
            p = (p + 1) % N
        seen.add(p)
        out.append(p)
    return np.array(out, dtype=int)

def _random_fixed_perm(N, rng):
    """Random but fixed (seeded) permutation."""
    return rng.permutation(N)

def _random_ortho_block(block_size, rng):
    """Generate a random orthogonal block of given size."""
    if block_size == 1:
        return np.array([[rng.choice([-1.0, 1.0])]])
    H = rng.standard_normal((block_size, block_size))
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q

def _make_block_diagonal_layer(N, block_size, rng):
    """Block-diagonal orthogonal matrix with given block size."""
    n_full_blocks = N // block_size
    remainder = N % block_size
    blocks = []
    for _ in range(n_full_blocks):
        blocks.append(_random_ortho_block(block_size, rng))
    if remainder > 0:
        blocks.append(_random_ortho_block(remainder, rng))
    return block_diag(*blocks)

def build_staged_orthogonal(N, L, block_size, perm_schedule, rng):
    """
    A = P_L B_L ... P_1 B_1
    Each B_l is block-diagonal orthogonal, each P_l is a permutation.
    """
    A = np.eye(N)
    layers_info = []
    for l in range(L):
        B = _make_block_diagonal_layer(N, block_size, rng)
        if l < len(perm_schedule):
            perm = perm_schedule[l]
        else:
            perm = np.arange(N)
        P = np.eye(N)[perm]
        A = P @ B @ A
        layers_info.append({"block": B.tolist(), "perm": perm.tolist()})
    return A, layers_info

def apply_staged_orthogonal(x, layers_info_np):
    """
    Factorised staged apply: apply B then P for each layer.
    layers_info_np: list of (B_blocks, perm) where B_blocks is list of (block, start_idx).
    """
    y = x.copy()
    for block_list, perm in layers_info_np:
        #apply block-diagonal
        for block, start in block_list:
            sz = block.shape[0]
            y[start:start+sz] = block @ y[start:start+sz]
        #apply permutation
        y = y[perm]
    return y

def _prepare_staged_apply_data(N, block_size, layers_info):
    """Convert layers_info to efficient numpy structures for apply."""
    result = []
    for layer in layers_info:
        B = np.array(layer["block"])
        perm = np.array(layer["perm"], dtype=int)
        #extract blocks
        blocks = []
        n_full = N // block_size
        remainder = N % block_size
        idx = 0
        for i in range(n_full):
            start = i * block_size
            blk = B[start:start+block_size, start:start+block_size]
            blocks.append((blk, start))
        if remainder > 0:
            start = n_full * block_size
            blk = B[start:start+remainder, start:start+remainder]
            blocks.append((blk, start))
        result.append((blocks, perm))
    return result

PERM_SCHEDULES = {
    "perfect_shuffle": lambda N, L, rng: [_perfect_shuffle_perm(N) for _ in range(L)],
    "bit_reversal": lambda N, L, rng: [_bit_reversal_perm(N) if l % 2 == 0
                                        else _perfect_shuffle_perm(N) for l in range(L)],
    "stride": lambda N, L, rng: [_stride_perm(N, stride=2+l) for l in range(L)],
    "random_fixed": lambda N, L, rng: [_random_fixed_perm(N, rng) for _ in range(L)],
}

def family_staged(N, L, block_size, perm_name, seed):
    """Generate a staged orthogonal instance."""
    rng = np.random.default_rng(seed)
    perm_fn = PERM_SCHEDULES[perm_name]
    perm_schedule = perm_fn(N, L, rng)
    #reset rng for block generation (permutation generation consumed some entropy)
    rng2 = np.random.default_rng(seed + 1000)
    A, layers_info = build_staged_orthogonal(N, L, block_size, perm_schedule, rng2)
    apply_data = _prepare_staged_apply_data(N, block_size, layers_info)
    params = {
        "seed": seed, "L": L, "block_size": block_size,
        "perm_name": perm_name,
    }
    return {
        "family": f"staged_b{block_size}_L{L}_{perm_name}",
        "N": N,
        "matrix": A,
        "params": params,
        "apply_fn": lambda x, _d=apply_data: apply_staged_orthogonal(x, _d),
        "apply_name": f"staged_b{block_size}_L{L}_factorised",
    }


#============================================================================
#ii. metrics
#============================================================================

def orthogonality_error(A):
    """||A^T A - I||_F"""
    N = A.shape[0]
    return float(np.linalg.norm(A.T @ A - np.eye(N), 'fro'))

def matrix_density(A, tol=1e-12):
    """Fraction of non-zero entries and nnz count."""
    nnz = int(np.sum(np.abs(A) > tol))
    total = A.shape[0] * A.shape[1]
    return nnz, nnz / total

def eigenvalue_analysis(A):
    """Eigenvalues, eigenangles, angular spread."""
    eigvals = np.linalg.eigvals(A)
    angles = np.angle(eigvals)
    magnitudes = np.abs(eigvals)
    return {
        "eigenvalues_real": np.real(eigvals).tolist(),
        "eigenvalues_imag": np.imag(eigvals).tolist(),
        "eigenangles": angles.tolist(),
        "eigenmagnitudes": magnitudes.tolist(),
        "angle_spread_std": float(np.std(angles)),
        "max_eigval_magnitude": float(np.max(magnitudes)),
    }

def build_state_matrix(A, delays, gamma):
    """Build expanded FDN state matrix F (M x M)."""
    N = A.shape[0]
    delays_arr = np.array(delays)
    M = int(np.sum(delays_arr))
    offsets = np.zeros(N, dtype=int)
    np.cumsum(delays_arr[:-1], out=offsets[1:])

    F = np.zeros((M, M))
    for i in range(N):
        o = offsets[i]
        m = delays[i]
        if m > 1:
            idx = np.arange(1, m)
            F[o + idx, o + idx - 1] = 1.0
    gamma_m = gamma ** delays_arr
    for i in range(N):
        for j in range(N):
            F[offsets[i], offsets[j] + delays[j] - 1] = A[i, j] * gamma_m[j]
    return F, offsets

def modal_std(A, delays, gamma):
    """
    Modal standard deviation: std(20*log10(|rho_k|)) in dB.
    Lower = more uniform = less coloration.
    Uses scipy.linalg.eig with left+right eigenvectors (no inv needed).
    """
    from scipy.linalg import eig as scipy_eig
    N = A.shape[0]
    F, offsets = build_state_matrix(A, delays, gamma)
    M = int(np.sum(delays))

    b = np.zeros(M)
    c = np.zeros(M)
    scale = 1.0 / np.sqrt(N)
    for i in range(N):
        b[offsets[i]] = scale
        c[offsets[i] + delays[i] - 1] = scale

    eigenvalues, V_left, V_right = scipy_eig(F, left=True, right=True)

    #residues: rho_k = (c . v_right_k) * (v_left_k^h . b)
    cv = c @ V_right
    ub = V_left.conj().T @ b
    residues = cv * ub

    magnitudes = np.abs(residues)
    threshold = np.max(magnitudes) * 1e-12
    valid = magnitudes > threshold
    if np.sum(valid) < 2:
        return 0.0, eigenvalues
    mag_db = 20.0 * np.log10(magnitudes[valid])
    return float(np.std(mag_db)), eigenvalues

def support_graph_metrics(A, tol=1e-12):
    """
    Graph diameter and average shortest path of the support graph.
    Support graph: edge (i,j) exists if |A[i,j]| > tol.
    """
    N = A.shape[0]
    adj = (np.abs(A) > tol).astype(float)
    #use unweighted shortest paths
    dist = shortest_path(adj, directed=False, unweighted=True)
    #handle disconnected graphs
    finite_mask = np.isfinite(dist) & (dist > 0)
    if not np.any(finite_mask):
        return {"diameter": float('inf'), "avg_shortest_path": float('inf'), "connected": False}
    diameter = float(np.max(dist[finite_mask]))
    avg_sp = float(np.mean(dist[finite_mask]))
    connected = bool(np.all(np.isfinite(dist)))
    return {"diameter": diameter, "avg_shortest_path": avg_sp, "connected": connected}


#============================================================================
#iii. fdn ir generation
#============================================================================

def generate_ir(A, delays, condition="lossless", apply_fn=None, fs=FS, duration=IR_DURATION):
    """
    Generate FDN impulse response.

    condition:
      "lossless"   : no decay, no filters (gamma=1, alpha=0)
      "attenuated" : matched T60 decay + one-pole lowpass

    If apply_fn is provided, uses factorised apply; else dense A@x.
    """
    N = A.shape[0]
    total_samples = int(duration * fs)

    if condition == "lossless":
        gamma = 1.0
        alpha = 0.0
    else:
        gamma = 10 ** (-3.0 / (T60_SECONDS * fs))
        alpha = ALPHA_LOWPASS

    gamma_m = gamma ** np.array(delays)

    buffers = [np.zeros(m) for m in delays]
    pointers = np.zeros(N, dtype=int)
    filter_states = np.zeros(N)

    b_vec = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]) / np.sqrt(N)
    c_vec = np.array([1.0 if (i // 2) % 2 == 0 else -1.0 for i in range(N)]) / np.sqrt(N)

    ir = np.zeros(total_samples)

    for n in range(total_samples):
        outs = np.array([buffers[i][pointers[i]] for i in range(N)])

        if alpha > 0:
            filter_states = (1 - alpha) * outs + alpha * filter_states
            filtered = filter_states
        else:
            filtered = outs

        ir[n] = np.dot(c_vec, filtered)

        in_val = 1.0 if n == 0 else 0.0
        feedback_input = filtered * gamma_m

        if apply_fn is not None:
            feedback = apply_fn(feedback_input) + b_vec * in_val
        else:
            feedback = A @ feedback_input + b_vec * in_val

        for i in range(N):
            buffers[i][pointers[i]] = feedback[i]
            pointers[i] = (pointers[i] + 1) % delays[i]

    return ir

def echo_density(ir, fs=FS, window_ms=10):
    """
    Echo density: count peaks per window over time.
    Returns times (s) and density (peaks/s) arrays.
    """
    window_samples = int(window_ms * fs / 1000)
    n_windows = len(ir) // window_samples
    times = np.zeros(n_windows)
    densities = np.zeros(n_windows)
    abs_ir = np.abs(ir)
    threshold = np.max(abs_ir) * 0.01
    if threshold < 1e-15:
        return times, densities

    for w in range(n_windows):
        start = w * window_samples
        end = start + window_samples
        chunk = abs_ir[start:end]
        peaks, _ = find_peaks(chunk, height=threshold * 0.1)
        times[w] = (start + window_samples // 2) / fs
        densities[w] = len(peaks) / (window_ms / 1000.0)

    return times, densities

def echo_density_summary(ir, fs=FS):
    """Summary statistics from echo density curve."""
    times, densities = echo_density(ir, fs)
    if len(densities) == 0:
        return {"mean_density": 0.0, "max_density": 0.0, "density_at_100ms": 0.0}
    idx_100ms = min(int(0.1 * fs / (10 * fs / 1000)), len(densities) - 1)
    return {
        "mean_density": float(np.mean(densities)),
        "max_density": float(np.max(densities)),
        "density_at_100ms": float(densities[idx_100ms]) if idx_100ms < len(densities) else 0.0,
    }


#============================================================================
#iv. runtime benchmarking
#============================================================================

def benchmark_apply(apply_fn, A, N, repeats=TIMING_REPEATS, warmup=TIMING_WARMUP):
    """
    Benchmark factorised apply vs dense multiply.
    Returns times in microseconds per call.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N)

    #warmup factorised
    for _ in range(warmup):
        apply_fn(x)

    #time factorised
    t0 = time.perf_counter()
    for _ in range(repeats):
        apply_fn(x)
    t_factorised = (time.perf_counter() - t0) / repeats * 1e6  #microseconds

    #warmup dense
    for _ in range(warmup):
        A @ x

    #time dense
    t0 = time.perf_counter()
    for _ in range(repeats):
        A @ x
    t_dense = (time.perf_counter() - t0) / repeats * 1e6  #microseconds

    return {
        "factorised_us": round(t_factorised, 3),
        "dense_us": round(t_dense, 3),
        "speedup": round(t_dense / max(t_factorised, 1e-9), 2),
    }


#============================================================================
#v. full evaluation pipeline
#============================================================================

def evaluate_instance(instance, delays, run_ir=True, run_timing=True, run_modal=True):
    """Run all metrics on a single family instance."""
    A = instance["matrix"]
    N = instance["N"]
    family = instance["family"]
    apply_fn = instance["apply_fn"]

    gamma = 10 ** (-3.0 / (T60_SECONDS * FS))

    result = {
        "family": family,
        "N": N,
        "apply_name": instance["apply_name"],
    }

    #store params (excluding large arrays for csv)
    result["param_seed"] = instance["params"].get("seed", -1)

    #orthogonality
    result["orth_error"] = orthogonality_error(A)

    #density
    nnz, density = matrix_density(A)
    result["nnz"] = nnz
    result["density"] = round(density, 4)

    #eigenvalue analysis
    eig_info = eigenvalue_analysis(A)
    result["angle_spread_std"] = eig_info["angle_spread_std"]
    result["max_eigval_mag"] = eig_info["max_eigval_magnitude"]

    #modal std (skip for large n where m=sum(delays) makes eigendecomp slow)
    if run_modal:
        ms, _ = modal_std(A, delays, gamma)
        result["modal_std_dB"] = round(ms, 4)
    else:
        result["modal_std_dB"] = float('nan')

    #graph metrics
    graph = support_graph_metrics(A)
    result["graph_diameter"] = graph["diameter"]
    result["avg_shortest_path"] = round(graph["avg_shortest_path"], 4)
    result["graph_connected"] = graph["connected"]

    #timing
    if run_timing:
        timing = benchmark_apply(apply_fn, A, N)
        result["factorised_us"] = timing["factorised_us"]
        result["dense_us"] = timing["dense_us"]
        result["speedup"] = timing["speedup"]

    #ir and echo density (both conditions)
    if run_ir:
        for cond in ["lossless", "attenuated"]:
            ir = generate_ir(A, delays, condition=cond, apply_fn=None)
            ed = echo_density_summary(ir)
            result[f"echo_mean_{cond}"] = round(ed["mean_density"], 1)
            result[f"echo_max_{cond}"] = round(ed["max_density"], 1)
            result[f"echo_100ms_{cond}"] = round(ed["density_at_100ms"], 1)
            result[f"ir_{cond}"] = ir  #stored for saving, not in csv

    #store explicit matrix for saving
    result["explicit_matrix"] = A

    return result


#============================================================================
#vi. family generator registry
#============================================================================

def generate_all_instances(N, seed):
    """Generate all family instances for a given N."""
    instances = []

    #dense orthogonal (3 seeds)
    for s in range(3):
        instances.append(family_dense_orthogonal(N, seed + s))

    #cmv (3 seeds)
    for s in range(3):
        instances.append(family_cmv(N, seed + 100 + s))

    #k-householder (k=1,2,3,4)
    for K in [1, 2, 3, 4]:
        for s in range(2):
            instances.append(family_k_householder(N, K, seed + 200 + K * 10 + s))

    #staged orthogonal
    for block_size in [2, 4]:
        for L in [2, 3, 4, 5, 6]:
            if block_size > N:
                continue
            for perm_name in ["perfect_shuffle", "bit_reversal", "stride", "random_fixed"]:
                instances.append(family_staged(N, L, block_size, perm_name, seed + 300 + block_size * 100 + L * 10))

    return instances


def generate_showdown_instances(seed):
    """
    Priority showdown: N=6, CMV vs Householder K=2 vs K=3 vs Dense.
    Plus staged-4x4 depth-3 and depth-4 if feasible.
    """
    N = 6
    instances = []

    #dense orthogonal, 5 seeds for statistics
    for s in range(5):
        instances.append(family_dense_orthogonal(N, seed + s))

    #cmv, 5 seeds
    for s in range(5):
        instances.append(family_cmv(N, seed + 100 + s))

    #householder k=2, 5 seeds
    for s in range(5):
        instances.append(family_k_householder(N, K=2, seed=seed + 200 + s))

    #householder k=3, 5 seeds
    for s in range(5):
        instances.append(family_k_householder(N, K=3, seed=seed + 300 + s))

    #staged 4x4 depth-3 (only if n>=4; for n=6 we get one 4x4 block + one 2x2 block)
    for perm_name in ["perfect_shuffle", "random_fixed"]:
        instances.append(family_staged(N, L=3, block_size=4, perm_name=perm_name, seed=seed + 400))

    #staged 4x4 depth-4
    for perm_name in ["perfect_shuffle", "random_fixed"]:
        instances.append(family_staged(N, L=4, block_size=4, perm_name=perm_name, seed=seed + 500))

    return instances


#============================================================================
#vii. output
#============================================================================

def save_results(all_results, output_dir):
    """Save master CSV/JSON table and per-family folders."""
    os.makedirs(output_dir, exist_ok=True)

    #--- master csv ---
    csv_fields = [
        "family", "N", "param_seed", "apply_name",
        "orth_error", "nnz", "density",
        "angle_spread_std", "max_eigval_mag",
        "modal_std_dB",
        "graph_diameter", "avg_shortest_path", "graph_connected",
        "factorised_us", "dense_us", "speedup",
        "echo_mean_lossless", "echo_max_lossless", "echo_100ms_lossless",
        "echo_mean_attenuated", "echo_max_attenuated", "echo_100ms_attenuated",
    ]

    csv_path = os.path.join(output_dir, "master_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            row = {k: v for k, v in r.items() if k in csv_fields}
            writer.writerow(row)

    #--- master json ---
    json_results = []
    for r in all_results:
        jr = {}
        for k, v in r.items():
            if k == "explicit_matrix":
                jr[k] = v.tolist()
            elif k.startswith("ir_"):
                jr[k] = v.tolist()
            elif isinstance(v, np.ndarray):
                jr[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                jr[k] = float(v)
            else:
                jr[k] = v
        json_results.append(jr)

    json_path = os.path.join(output_dir, "master_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=1)

    #--- per-family folders ---
    families = set(r["family"] for r in all_results)
    for fam in families:
        fam_dir = os.path.join(output_dir, fam)
        os.makedirs(fam_dir, exist_ok=True)

        fam_results = [r for r in all_results if r["family"] == fam]
        for i, r in enumerate(fam_results):
            N = r["N"]
            seed = r.get("param_seed", i)
            prefix = f"N{N}_s{seed}"

            #save matrix
            np.save(os.path.join(fam_dir, f"{prefix}_matrix.npy"), r["explicit_matrix"])

            #save irs
            for cond in ["lossless", "attenuated"]:
                ir_key = f"ir_{cond}"
                if ir_key in r and r[ir_key] is not None:
                    np.save(os.path.join(fam_dir, f"{prefix}_ir_{cond}.npy"), r[ir_key])

                    #small wav render
                    try:
                        import scipy.io.wavfile as wavfile
                        ir_data = r[ir_key]
                        peak = np.max(np.abs(ir_data))
                        if peak > 0:
                            ir_norm = (ir_data / peak * 0.9 * 32767).astype(np.int16)
                        else:
                            ir_norm = np.zeros_like(ir_data, dtype=np.int16)
                        wavfile.write(
                            os.path.join(fam_dir, f"{prefix}_ir_{cond}.wav"),
                            FS, ir_norm
                        )
                    except Exception:
                        pass

    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Families: {sorted(families)}")
    return csv_path, json_path


#============================================================================
#viii. main: showdown first, then broad sweep
#============================================================================

def run_showdown(output_dir):
    """
    Priority showdown: N=6
    CMV vs Householder K=2 vs K=3 vs Dense orthogonal
    + staged-4x4 depth-3/4
    """
    print("=" * 70)
    print("  PRIORITY SHOWDOWN: N=6")
    print("=" * 70)

    delays = DELAY_SETS[6]
    instances = generate_showdown_instances(MASTER_SEED)

    results = []
    for inst in instances:
        print(f"  Evaluating {inst['family']:30s} (seed={inst['params'].get('seed', '?')}) ...", end="", flush=True)
        r = evaluate_instance(inst, delays, run_ir=True, run_timing=True)
        results.append(r)
        print(f"  modal_std={r['modal_std_dB']:7.2f} dB  "
              f"factorised={r.get('factorised_us', 'N/A')} us  "
              f"diameter={r['graph_diameter']}")

    showdown_dir = os.path.join(output_dir, "_showdown_N6")
    save_results(results, showdown_dir)

    #print summary table
    print()
    print("  SHOWDOWN SUMMARY (N=6)")
    print("  " + "-" * 90)
    print(f"  {'Family':35s} {'modal_std':>10s} {'orth_err':>10s} {'diam':>6s} {'avgSP':>7s} "
          f"{'fact_us':>8s} {'dense_us':>8s} {'speedup':>7s}")
    print("  " + "-" * 90)

    #group by family and average
    from collections import defaultdict
    by_family = defaultdict(list)
    for r in results:
        by_family[r["family"]].append(r)

    for fam in sorted(by_family.keys()):
        rs = by_family[fam]
        avg_ms = np.mean([r["modal_std_dB"] for r in rs])
        avg_oe = np.mean([r["orth_error"] for r in rs])
        diam = rs[0]["graph_diameter"]
        avg_sp = np.mean([r["avg_shortest_path"] for r in rs])
        avg_fu = np.mean([r.get("factorised_us", 0) for r in rs])
        avg_du = np.mean([r.get("dense_us", 0) for r in rs])
        spd = avg_du / max(avg_fu, 1e-9)
        print(f"  {fam:35s} {avg_ms:10.2f} {avg_oe:10.2e} {diam:6.0f} {avg_sp:7.2f} "
              f"{avg_fu:8.1f} {avg_du:8.1f} {spd:7.2f}")

    print()
    return results


def run_broad_sweep(output_dir, sizes=None):
    """Run all families across specified sizes."""
    if sizes is None:
        sizes = SIZES_PRIMARY

    print("=" * 70)
    print(f"  BROAD SWEEP: N = {sizes}")
    print("=" * 70)

    all_results = []
    for N in sizes:
        if N not in DELAY_SETS:
            print(f"  Skipping N={N} (no delay set defined)")
            continue

        delays = DELAY_SETS[N]
        print(f"\n  --- N = {N} (delays: {delays[:4]}...) ---")

        instances = generate_all_instances(N, MASTER_SEED)

        #for large n, skip ir generation and modal_std (slow eigendecomp)
        run_ir = N <= 8
        run_modal = N <= 16

        for inst in instances:
            fam = inst["family"]
            print(f"    {fam:40s}", end="", flush=True)
            try:
                r = evaluate_instance(inst, delays, run_ir=run_ir, run_timing=True, run_modal=run_modal)
                all_results.append(r)
                ms_str = f"{r['modal_std_dB']:7.2f}" if not np.isnan(r['modal_std_dB']) else "    N/A"
                print(f"  modal_std={ms_str}  diam={r['graph_diameter']:.0f}  "
                      f"fact={r.get('factorised_us', 'N/A')} us")
            except Exception as e:
                print(f"  ERROR: {e}")

    sweep_dir = os.path.join(output_dir, "_broad_sweep")
    save_results(all_results, sweep_dir)
    return all_results


def main():
    print()
    print("*" * 70)
    print("  ORTHOGONAL FDN FEEDBACK FAMILY BENCHMARK")
    print("  Experiment 21 — Reproducible Harness")
    print("*" * 70)
    print()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_output")
    os.makedirs(output_dir, exist_ok=True)

    #priority showdown (n=6)
    showdown_results = run_showdown(output_dir)

    #broad sweep (primary sizes)
    broad_results = run_broad_sweep(output_dir, SIZES_PRIMARY)

    #optional large sizes (timing and graph metrics only, no ir or modal_std)
    print("\n  Running optional large sizes (timing + graph only)...")
    for N in SIZES_OPTIONAL:
        if N not in DELAY_SETS:
            continue
        delays = DELAY_SETS[N]
        print(f"\n  --- N = {N} ---")
        for fam_fn, kwargs_list in [
            (family_dense_orthogonal, [{"N": N, "seed": MASTER_SEED}]),
            (family_cmv, [{"N": N, "seed": MASTER_SEED + 100}]),
            (family_k_householder, [{"N": N, "K": 2, "seed": MASTER_SEED + 200}]),
            (family_k_householder, [{"N": N, "K": 3, "seed": MASTER_SEED + 300}]),
        ]:
            for kw in kwargs_list:
                inst = fam_fn(**kw)
                print(f"    {inst['family']:40s}", end="", flush=True)
                try:
                    r = evaluate_instance(inst, delays, run_ir=False, run_timing=True, run_modal=False)
                    print(f"  diam={r['graph_diameter']:.0f}  fact={r.get('factorised_us', 'N/A')} us  "
                          f"dense={r.get('dense_us', 'N/A')} us")
                except Exception as e:
                    print(f"  ERROR: {e}")

    print()
    print("=" * 70)
    print("  BENCHMARK COMPLETE")
    print(f"  Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
