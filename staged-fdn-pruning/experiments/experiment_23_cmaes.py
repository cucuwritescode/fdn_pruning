"""
Experiment 23: CMA-ES Optimisation (Protocol-Compliant)
========================================================
Follows PROTOCOL_FAMILY_BENCHMARK.md v1.1.

Phase 1 priority:
  1. Staged b=2 L=3 perfect_shuffle, N=16, CMA-ES, 5 restarts
  2. CMV N=16, 2 additional restarts (supplement exp22's 3)
  3. Staged b=2 L=3 stride, N=16, CMA-ES, 5 restarts

CMA-ES config: sigma0=pi/3, budget=200*d per restart, bounds [0, 2*pi].

Author: Facundo Franchino
"""

import numpy as np
import cma
import json
import os
import time
from scipy.linalg import eig as scipy_eig, block_diag

#============================================================================
#constants
#============================================================================

MASTER_SEED = 42
FS = 48000
T60_SECONDS = 2.0

#short delays for fast optimisation (m~30-560 vs 480-3000)
#rank correlation with long delays: spearman rho=0.87, p=0.001
DELAY_SETS_OPT = {
    6:  [7, 11, 13, 17, 19, 23],
    8:  [7, 11, 13, 17, 19, 23, 29, 31],
    16: [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67],
}

#long delays for final validation
DELAY_SETS_VAL = {
    6:  [97, 113, 127, 139, 151, 163],
    8:  [97, 113, 127, 139, 151, 163, 173, 191],
    16: [97, 113, 127, 139, 151, 163, 173, 191, 199, 211, 223, 233, 241, 251, 263, 269],
}


#============================================================================
#modal std (scipy.linalg.eig, left+right)
#============================================================================

def modal_std(A, delays, gamma):
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

    b = np.zeros(M)
    c = np.zeros(M)
    scale = 1.0 / np.sqrt(N)
    for i in range(N):
        b[offsets[i]] = scale
        c[offsets[i] + delays[i] - 1] = scale

    eigenvalues, V_left, V_right = scipy_eig(F, left=True, right=True)
    cv = c @ V_right
    ub = V_left.conj().T @ b
    residues = cv * ub
    magnitudes = np.abs(residues)
    threshold = np.max(magnitudes) * 1e-12
    valid = magnitudes > threshold
    if np.sum(valid) < 2:
        return 0.0
    mag_db = 20.0 * np.log10(magnitudes[valid])
    return float(np.std(mag_db))


#============================================================================
#cmv
#============================================================================

def build_cmv(thetas, N):
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


#============================================================================
#staged orthogonal
#============================================================================

def _perfect_shuffle_perm(N):
    half = N // 2
    perm = np.zeros(N, dtype=int)
    for i in range(half):
        perm[2*i] = i
        perm[2*i+1] = half + i
    if N % 2 == 1:
        perm[N-1] = N-1
    return perm

def _stride_perm(N, stride=2):
    perm = np.array([(i * stride) % N for i in range(N)], dtype=int)
    seen = set()
    out = []
    for p in perm:
        while p in seen:
            p = (p + 1) % N
        seen.add(p)
        out.append(p)
    return np.array(out, dtype=int)

def _angles_to_ortho_2x2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def _angles_to_ortho_4x4(angles):
    A = np.eye(4)
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for k, (i, j) in enumerate(pairs):
        c, s = np.cos(angles[k]), np.sin(angles[k])
        G = np.eye(4)
        G[i,i] = c; G[i,j] = -s
        G[j,i] = s; G[j,j] = c
        A = G @ A
    return A

def _angles_to_ortho_block(angles, block_size):
    if block_size == 1:
        return np.array([[1.0]])
    elif block_size == 2:
        return _angles_to_ortho_2x2(angles[0])
    elif block_size == 4:
        return _angles_to_ortho_4x4(angles)
    else:
        n = block_size
        A = np.eye(n)
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        for k, (i, j) in enumerate(pairs):
            if k >= len(angles):
                break
            c, s = np.cos(angles[k]), np.sin(angles[k])
            G = np.eye(n)
            G[i,i] = c; G[i,j] = -s
            G[j,i] = s; G[j,j] = c
            A = G @ A
        return A

def build_staged_from_params(x, N, L, block_size, perm_schedule):
    n_full = N // block_size
    remainder = N % block_size
    angles_per_block = block_size * (block_size - 1) // 2
    rem_angles = remainder * (remainder - 1) // 2

    A = np.eye(N)
    idx = 0
    for l in range(L):
        blocks = []
        for b in range(n_full):
            angles = x[idx:idx+angles_per_block]
            idx += angles_per_block
            blocks.append(_angles_to_ortho_block(angles, block_size))
        if remainder > 0:
            angles = x[idx:idx+rem_angles]
            idx += rem_angles
            blocks.append(_angles_to_ortho_block(angles, remainder))
        B = block_diag(*blocks)
        P = np.eye(N)[perm_schedule[l]]
        A = P @ B @ A
    return A

def count_staged_params(N, L, block_size):
    n_full = N // block_size
    remainder = N % block_size
    angles_per_layer = n_full * (block_size * (block_size - 1) // 2)
    if remainder > 0:
        angles_per_layer += remainder * (remainder - 1) // 2
    return L * angles_per_layer


#============================================================================
#cma-es optimiser
#============================================================================

def optimise_cmaes(objective, d, n_restarts=5, budget_per_restart=None, label=""):
    """Run CMA-ES with multiple restarts. Returns (best_x, best_val, all_restart_vals)."""
    if budget_per_restart is None:
        budget_per_restart = max(50 * d, 500)

    sigma0 = np.pi / 3
    best_x = None
    best_val = float('inf')
    all_vals = []
    rng = np.random.default_rng(MASTER_SEED)

    for restart in range(n_restarts):
        x0 = rng.uniform(0, 2 * np.pi, d)
        print(f"    {label} restart {restart+1}/{n_restarts}...", end="", flush=True)
        t0 = time.time()

        opts = {
            'maxfevals': budget_per_restart,
            'bounds': [0, 2 * np.pi],
            'seed': MASTER_SEED + restart * 77,
            'verbose': -9,  #suppress cma output
            'tolfun': 1e-3,
        }

        res = cma.fmin2(objective, x0, sigma0, opts)
        xbest = res[0]
        val = res[1].result.fbest

        elapsed = time.time() - t0
        n_evals = res[1].result.evaluations
        print(f" modal_std={val:.4f} dB ({n_evals} evals, {elapsed:.1f}s)")

        all_vals.append(val)
        if val < best_val:
            best_val = val
            best_x = xbest

    return best_x, best_val, all_vals


#============================================================================
#controls
#============================================================================

def gen_dense_orthogonal(N, seed):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((N, N))
    Q, R = np.linalg.qr(H)
    return Q @ np.diag(np.sign(np.diag(R)))

def build_k_householder(N, K, seed):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((K, N))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    A = np.eye(N)
    for v in vectors:
        A = (np.eye(N) - 2.0 * np.outer(v, v)) @ A
    return A

def evaluate_controls(N, delays, n_samples=5):
    gamma = 10 ** (-3.0 / (T60_SECONDS * FS))
    results = {}
    for name, gen_fn in [
        ("dense_orthogonal", lambda s: gen_dense_orthogonal(N, s)),
        ("householder_K2", lambda s: build_k_householder(N, 2, s)),
        ("householder_K3", lambda s: build_k_householder(N, 3, s)),
    ]:
        vals = [modal_std(gen_fn(MASTER_SEED + s), delays, gamma) for s in range(n_samples)]
        results[name] = {
            "values": vals,
            "best": float(np.min(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    return results


#============================================================================
#main
#============================================================================

def run_sweep(N, delays_opt, delays_val, n_restarts=5, output_dir="."):
    """Run CMA-ES sweep for one N: optimise on short delays, validate on long."""
    gamma = 10 ** (-3.0 / (T60_SECONDS * FS))
    results = {"N": N}

    print(f"\n{'='*70}")
    print(f"  N={N}  (opt M={sum(delays_opt)}, val M={sum(delays_val)})")
    print(f"{'='*70}")

    #controls (evaluated on both delay sets)
    print("\n  Controls (opt delays)...")
    controls = evaluate_controls(N, delays_opt)
    for name, ctrl in controls.items():
        print(f"    {name:25s}  best={ctrl['best']:.2f}  mean={ctrl['mean']:.2f}")
    results["controls_opt"] = controls

    configs = []

    #cmv
    d_cmv = N - 1
    def obj_cmv(x):
        return modal_std(build_cmv(x, N), delays_opt, gamma)
    configs.append(("cmv", obj_cmv, d_cmv, lambda x: build_cmv(x, N)))

    #staged b=2 l=3, perfect_shuffle
    perm_ps = [_perfect_shuffle_perm(N) for _ in range(3)]
    d_s2 = count_staged_params(N, 3, 2)
    def obj_s2ps(x):
        return modal_std(build_staged_from_params(x, N, 3, 2, perm_ps), delays_opt, gamma)
    configs.append(("staged_b2_L3_PS", obj_s2ps, d_s2,
                     lambda x: build_staged_from_params(x, N, 3, 2, perm_ps)))

    #staged b=2 l=3, stride
    perm_st = [_stride_perm(N) for _ in range(3)]
    def obj_s2st(x):
        return modal_std(build_staged_from_params(x, N, 3, 2, perm_st), delays_opt, gamma)
    configs.append(("staged_b2_L3_stride", obj_s2st, d_s2,
                     lambda x: build_staged_from_params(x, N, 3, 2, perm_st)))

    #run each
    for name, obj, d, build_fn in configs:
        print(f"\n  {name} (d={d})...")
        x_opt, val_opt, restart_vals = optimise_cmaes(obj, d, n_restarts=n_restarts, label=name)

        #validate best on long delays
        A_best = build_fn(x_opt)
        val_val = modal_std(A_best, delays_val, gamma)
        orth_err = float(np.linalg.norm(A_best.T @ A_best - np.eye(N)))

        results[name] = {
            "best_opt": val_opt, "best_val": val_val,
            "restarts_opt": restart_vals,
            "mean_opt": float(np.mean(restart_vals)),
            "std_opt": float(np.std(restart_vals)),
            "params": x_opt.tolist(), "d": d,
            "orth_error": orth_err,
        }
        print(f"    >>> opt={val_opt:.4f} dB  validated={val_val:.4f} dB  orth_err={orth_err:.2e}")

    #summary
    print()
    print("=" * 70)
    print(f"  SUMMARY N={N}")
    print("=" * 70)
    print(f"  {'Config':30s} {'opt_best':>9s} {'val_best':>9s} {'opt_mean':>9s} {'std':>6s} {'d':>4s}")
    print("  " + "-" * 70)
    for name, _, d, _ in configs:
        r = results[name]
        print(f"  {name:30s} {r['best_opt']:9.2f} {r['best_val']:9.2f} "
              f"{r['mean_opt']:9.2f} {r['std_opt']:6.2f} {d:4d}")

    return results


def main():
    print()
    print("*" * 70)
    print("  EXPERIMENT 23: CMA-ES OPTIMISATION")
    print("  Short delays for optimisation, long delays for validation")
    print("  Rank correlation: Spearman rho=0.87, p=0.001")
    print("*" * 70)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmaes_output")
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for N in [6, 8, 16]:
        results = run_sweep(N, DELAY_SETS_OPT[N], DELAY_SETS_VAL[N],
                           n_restarts=5, output_dir=output_dir)
        all_results[f"N{N}"] = results

    #save
    def to_serializable(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict): return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_serializable(x) for x in obj]
        return obj

    json_path = os.path.join(output_dir, "cmaes_results.json")
    with open(json_path, "w") as f:
        json.dump(to_serializable(all_results), f, indent=2)
    print(f"\n  Saved to {json_path}")

    #cross-n comparison
    print()
    print("=" * 70)
    print("  CROSS-N COMPARISON (validated modal_std)")
    print("=" * 70)
    print(f"  {'Config':30s} {'N=6':>8s} {'N=8':>8s} {'N=16':>8s}")
    print("  " + "-" * 58)
    for cfg in ["cmv", "staged_b2_L3_PS", "staged_b2_L3_stride"]:
        line = f"  {cfg:30s}"
        for N in [6, 8, 16]:
            r = all_results[f"N{N}"].get(cfg, {})
            v = r.get("best_val", float('nan'))
            line += f" {v:8.2f}"
        print(line)

    print("\n  COMPLETE")


if __name__ == "__main__":
    main()
