"""
Experiment 25: Staged Projection Validation
=============================================
Validates the dense to staged b=2 PS projection method:
  1. Approximation quality (Frobenius error, relative trace) vs L and N
  2. Modal std preservation (colorlessness)
  3. Echo density / mixing time preservation (smoothness)
  4. Spectral flatness comparison
  5. Full pipeline demo: dense to project to evaluate

Author: Facundo Franchino
"""

import numpy as np
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(__file__))

from staged_fdn.projection import (
    project_dense_to_staged,
    build_staged_from_angles,
    apply_staged,
    perfect_shuffle_perm,
    perm_matrix,
)
from experiment_02_structured_validation import (
    build_state_matrix,
    build_input_output_vectors,
    modal_decomposition,
    modal_std,
)
from experiment_24_echo_density import (
    generate_ir,
    echo_density_curve,
    mixing_time,
    spectral_flatness_over_time,
    build_staged_b2_L3_PS,
    build_cmv,
    gen_dense_orthogonal,
    FS, T60, IR_DURATION,
    DELAY_SETS,
)


GAMMA = 0.9999
SEED = 42


#============================================================================
#approximation quality: frobenius error and relative trace vs l and n
#============================================================================

def experiment_approximation_quality():
    """How well does the staged manifold approximate dense orthogonal matrices?"""
    print("=" * 70)
    print("EXPERIMENT 25.1: Approximation Quality vs L and N")
    print("=" * 70)

    results = {}
    rng = np.random.default_rng(SEED)

    for N in [8, 16]:
        results[N] = {}
        #generate a dense orthogonal matrix
        A = gen_dense_orthogonal(N, SEED)

        for L in [3, 4, 5, 6]:
            res = project_dense_to_staged(A, N, L=L, n_restarts=20, seed=SEED)
            results[N][L] = {
                'rel_trace': res['rel_trace'],
                'frob_error': res['frob_error'],
                'n_params': L * (N // 2),
                'n_dof': N * (N - 1) // 2,
            }
            ratio = results[N][L]['n_params'] / results[N][L]['n_dof']
            print(f"  N={N:2d}, L={L}: rel_trace={res['rel_trace']:.4f}, "
                  f"||A-S||_F={res['frob_error']:.3f}, "
                  f"params/DOF={ratio:.2f}")
        print()

    return results


#============================================================================
#modal std: dense vs projected
#============================================================================

def experiment_modal_std_preservation():
    """Does projection preserve colorlessness (modal_std)?"""
    print("=" * 70)
    print("EXPERIMENT 25.2: Modal Std Preservation (Dense vs Projected)")
    print("=" * 70)

    results = {}
    for N in [8, 16]:
        delays = DELAY_SETS[N][:N] if N in DELAY_SETS else list(range(7, 7 + 2 * N, 2))[:N]

        results[N] = {}

        #dense orthogonal
        A_dense = gen_dense_orthogonal(N, SEED)
        sigma_dense, _, _ = modal_std(A_dense, delays, GAMMA)
        results[N]['dense'] = sigma_dense
        print(f"\n  N={N}: dense modal_std = {sigma_dense:.2f} dB")

        #project to staged for various l
        for L in [3, 4, 5, 6]:
            res = project_dense_to_staged(A_dense, N, L=L, n_restarts=20, seed=SEED)
            S = res['S']
            sigma_staged, _, _ = modal_std(S, delays, GAMMA)
            results[N][f'staged_L{L}'] = sigma_staged
            delta = sigma_staged - sigma_dense
            print(f"  N={N}, L={L}: staged modal_std = {sigma_staged:.2f} dB "
                  f"(Δ = {delta:+.2f} dB), rel_trace = {res['rel_trace']:.4f}")

        #random staged (baseline, no projection, just random angles)
        rng = np.random.default_rng(SEED + 100)
        rand_params = rng.uniform(0, 2 * np.pi, 3 * (N // 2))
        S_rand = build_staged_b2_L3_PS(rand_params, N)
        sigma_rand, _, _ = modal_std(S_rand, delays, GAMMA)
        results[N]['staged_random'] = sigma_rand
        print(f"  N={N}: random staged L=3 modal_std = {sigma_rand:.2f} dB (baseline)")

    return results


#============================================================================
#echo density and mixing time
#============================================================================

def experiment_echo_density_preservation():
    """Does projection preserve echo density and mixing time?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 25.3: Echo Density & Mixing Time (Dense vs Projected)")
    print("=" * 70)

    results = {}
    for N in [8, 16]:
        delays = DELAY_SETS[N][:N] if N in DELAY_SETS else list(range(7, 7 + 2 * N, 2))[:N]
        results[N] = {}

        #dense
        A_dense = gen_dense_orthogonal(N, SEED)
        ir_dense = generate_ir(A_dense, delays, condition="lossless")
        mt_dense = mixing_time(ir_dense)
        _, ed_dense = echo_density_curve(ir_dense)
        ed_median_dense = np.median(ed_dense[len(ed_dense) // 2:])
        results[N]['dense'] = {'mixing_time': mt_dense, 'ed_median': ed_median_dense}
        print(f"\n  N={N} dense: mixing_time={mt_dense:.1f}ms, "
              f"echo_density_median={ed_median_dense:.3f}")

        #projected staged for various l
        for L in [3, 4, 5, 6]:
            res = project_dense_to_staged(A_dense, N, L=L, n_restarts=20, seed=SEED)
            S = res['S']
            ir_staged = generate_ir(S, delays, condition="lossless")
            mt_staged = mixing_time(ir_staged)
            _, ed_staged = echo_density_curve(ir_staged)
            ed_median_staged = np.median(ed_staged[len(ed_staged) // 2:])
            results[N][f'staged_L{L}'] = {
                'mixing_time': mt_staged,
                'ed_median': ed_median_staged,
                'rel_trace': res['rel_trace'],
            }
            print(f"  N={N}, L={L}: mixing_time={mt_staged:.1f}ms, "
                  f"echo_density_median={ed_median_staged:.3f}, "
                  f"rel_trace={res['rel_trace']:.4f}")

        #random staged baseline
        rng = np.random.default_rng(SEED + 100)
        rand_params = rng.uniform(0, 2 * np.pi, 3 * (N // 2))
        S_rand = build_staged_b2_L3_PS(rand_params, N)
        ir_rand = generate_ir(S_rand, delays, condition="lossless")
        mt_rand = mixing_time(ir_rand)
        _, ed_rand = echo_density_curve(ir_rand)
        ed_median_rand = np.median(ed_rand[len(ed_rand) // 2:])
        results[N]['staged_random'] = {'mixing_time': mt_rand, 'ed_median': ed_median_rand}
        print(f"  N={N} random staged L=3: mixing_time={mt_rand:.1f}ms, "
              f"echo_density_median={ed_median_rand:.3f} (baseline)")

    return results


#============================================================================
#full pipeline: train dense, project, compare
#============================================================================

def experiment_full_pipeline():
    """Complete projection pipeline with multiple source matrices."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 25.4: Full Projection Pipeline")
    print("=" * 70)

    N = 16
    L = 3
    delays = DELAY_SETS[N]

    sources = {
        'dense_random': gen_dense_orthogonal(N, SEED),
        'dense_random_2': gen_dense_orthogonal(N, SEED + 1),
        'dense_random_3': gen_dense_orthogonal(N, SEED + 2),
    }

    #also add cmv as a source (known to work well for fdn)
    rng = np.random.default_rng(SEED)
    cmv_thetas = rng.uniform(0, 2 * np.pi, N - 1)
    sources['cmv'] = build_cmv(cmv_thetas, N)

    print(f"\n  Configuration: N={N}, L={L}, delays={delays}")
    print(f"  {'Source':<18s} {'modal_std_src':>13s} {'modal_std_proj':>14s} "
          f"{'Δ_dB':>7s} {'rel_trace':>10s} {'||A-S||_F':>10s}")
    print("  " + "-" * 75)

    results = []
    for name, A in sources.items():
        sigma_src, _, _ = modal_std(A, delays, GAMMA)
        res = project_dense_to_staged(A, N, L=L, n_restarts=25, seed=SEED)
        S = res['S']
        sigma_proj, _, _ = modal_std(S, delays, GAMMA)
        delta = sigma_proj - sigma_src
        results.append({
            'name': name,
            'sigma_src': sigma_src,
            'sigma_proj': sigma_proj,
            'delta': delta,
            'rel_trace': res['rel_trace'],
            'frob_error': res['frob_error'],
        })
        print(f"  {name:<18s} {sigma_src:>13.2f} {sigma_proj:>14.2f} "
              f"{delta:>+7.2f} {res['rel_trace']:>10.4f} {res['frob_error']:>10.3f}")

    return results


#============================================================================
#l sweep with echo density for the key n=16 case
#============================================================================

def experiment_L_sweep_N16():
    """Detailed L sweep for N=16 showing all metrics."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 25.5: L Sweep for N=16 (All Metrics)")
    print("=" * 70)

    N = 16
    delays = DELAY_SETS[N]
    A_dense = gen_dense_orthogonal(N, SEED)

    #dense baseline
    sigma_dense, _, _ = modal_std(A_dense, delays, GAMMA)
    ir_dense = generate_ir(A_dense, delays, condition="lossless")
    mt_dense = mixing_time(ir_dense)
    _, ed_dense = echo_density_curve(ir_dense)
    ed_median_dense = np.median(ed_dense[len(ed_dense) // 2:])

    print(f"\n  Dense baseline: modal_std={sigma_dense:.2f}dB, "
          f"mixing_time={mt_dense:.1f}ms, echo_density={ed_median_dense:.3f}")

    print(f"\n  {'L':>3s} {'rel_trace':>10s} {'modal_std':>10s} {'Δ_dB':>7s} "
          f"{'mix_time':>10s} {'echo_dens':>10s} {'params':>7s}")
    print("  " + "-" * 60)

    results = []
    for L in [3, 4, 5, 6, 7, 8]:
        res = project_dense_to_staged(A_dense, N, L=L, n_restarts=20, seed=SEED)
        S = res['S']
        sigma_s, _, _ = modal_std(S, delays, GAMMA)
        ir_s = generate_ir(S, delays, condition="lossless")
        mt_s = mixing_time(ir_s)
        _, ed_s = echo_density_curve(ir_s)
        ed_median_s = np.median(ed_s[len(ed_s) // 2:])

        n_params = L * (N // 2)
        delta = sigma_s - sigma_dense
        results.append({
            'L': L, 'rel_trace': res['rel_trace'],
            'sigma': sigma_s, 'delta': delta,
            'mixing_time': mt_s, 'echo_density': ed_median_s,
            'n_params': n_params,
        })
        print(f"  {L:3d} {res['rel_trace']:>10.4f} {sigma_s:>10.2f} {delta:>+7.2f} "
              f"{mt_s:>10.1f} {ed_median_s:>10.3f} {n_params:>7d}")

    return results


#============================================================================
#main
#============================================================================

if __name__ == "__main__":
    print("Experiment 25: Staged Projection Validation")
    print("=" * 70)

    r1 = experiment_approximation_quality()
    r2 = experiment_modal_std_preservation()
    r3 = experiment_echo_density_preservation()
    r4 = experiment_full_pipeline()
    r5 = experiment_L_sweep_N16()

    #save results
    out_dir = os.path.join(os.path.dirname(__file__), 'projection_output')
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        'approximation_quality': {str(k): {str(l): v for l, v in d.items()}
                                  for k, d in r1.items()},
        'full_pipeline': r4,
        'L_sweep_N16': r5,
    }

    with open(os.path.join(out_dir, 'experiment_25_results.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {out_dir}/experiment_25_results.json")
    print("\nExperiment 25 complete.")
