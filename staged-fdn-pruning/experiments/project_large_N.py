"""
Project trained N=128 and N=256 at multiple L values.
Generates IRs for listening comparison.

Author: Facundo Franchino, 2026
"""

import sys, os
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from staged_fdn.projection import (
    project_dense_to_staged, refine_with_lsd,
    fdn_frequency_response, log_spectral_distance,
    build_staged_from_angles, perm_matrix
)


def render_ir(A, delays, gamma=0.9999, b=None, c=None, duration_s=2.0, fs=48000):
    """Render FDN impulse response in the time domain."""
    N = A.shape[0]
    delays = np.asarray(delays, dtype=int)
    n_samples = int(duration_s * fs)

    if b is None:
        b = np.array([(-1.0)**i for i in range(N)]) / np.sqrt(N)
    if c is None:
        c = np.array([(-1.0)**(i // 2) for i in range(N)]) / np.sqrt(N)

    gamma_m = gamma ** delays

    #one flat ring buffer for all delay lines, per-line offsets into it
    offsets = np.concatenate([[0], np.cumsum(delays[:-1])])
    buf = np.zeros(int(delays.sum()))
    ptrs = np.zeros(N, dtype=int)
    idx = offsets + ptrs
    ir = np.zeros(n_samples)

    for n in range(n_samples):
        #read every delay line head in one gather
        state = buf[idx]

        #output
        ir[n] = np.dot(c, state)

        #feedback through the attenuated state
        fb = A @ (state * gamma_m)

        #input (impulse at n=0)
        if n == 0:
            fb = fb + b

        #write back at the read positions, then advance every pointer
        buf[idx] = fb
        ptrs += 1
        np.mod(ptrs, delays, out=ptrs)
        np.add(offsets, ptrs, out=idx)

    #normalise
    peak = np.max(np.abs(ir))
    if peak > 0:
        ir = ir / peak * 0.9

    return ir


def run_projection(N, L_values):
    print(f"\n{'='*60}")
    print(f"Projecting N={N}")
    print(f"{'='*60}")

    A = np.load(os.path.join(os.path.dirname(__file__), f'flamo_trained_N{N}.npy'))
    B = np.load(os.path.join(os.path.dirname(__file__), f'flamo_trained_N{N}_B.npy'))
    C = np.load(os.path.join(os.path.dirname(__file__), f'flamo_trained_N{N}_C.npy'))
    delays = np.load(os.path.join(os.path.dirname(__file__), f'flamo_trained_N{N}_delays.npy'))

    #verify orthogonality
    err = np.linalg.norm(A.T @ A - np.eye(N))
    print(f"  ||A^T A - I|| = {err:.2e}")

    gamma = 0.999
    out_dir = os.path.join(os.path.dirname(__file__), 'listening_L_comparison')
    os.makedirs(out_dir, exist_ok=True)

    #render dense ir
    print(f"  Rendering dense IR...")
    ir_dense = render_ir(A, delays, gamma=gamma, b=B, c=C)
    sf.write(os.path.join(out_dir, f'N{N}_dense_trained.wav'), ir_dense, 48000)

    results = []
    for L in L_values:
        print(f"\n  L={L}:")

        #phase 1 and 2: frobenius projection
        proj = project_dense_to_staged(A, N, L=L, n_restarts=15,
                                        max_iter=30, verbose=False)
        S_frob = proj['S']
        lsd_frob = log_spectral_distance(A, S_frob, delays, gamma=gamma,
                                          b=B, c=C, n_freqs=512)
        print(f"    Frobenius LSD: {lsd_frob:.2f} dB")

        #phase 3: lsd refinement
        refined = refine_with_lsd(proj, A, delays, gamma=gamma, b=B, c=C,
                                   n_freqs=512, maxiter=300, lr=5e-3,
                                   verbose=True)
        lsd_ref = refined['lsd']
        print(f"    Refined LSD:   {lsd_ref:.2f} dB")

        #render staged ir
        S_final = refined['S']
        ir_staged = render_ir(S_final, delays, gamma=gamma, b=B, c=C)
        sf.write(os.path.join(out_dir, f'N{N}_L{L}_trained.wav'), ir_staged, 48000)

        results.append({'L': L, 'lsd_frob': lsd_frob, 'lsd_refined': lsd_ref})

    #summary
    print(f"\n  --- N={N} Summary ---")
    print(f"  {'L':>3}  {'LSD Frob':>10}  {'LSD Refined':>12}")
    for r in results:
        print(f"  {r['L']:>3}  {r['lsd_frob']:>10.2f}  {r['lsd_refined']:>12.2f}")

    return results


if __name__ == '__main__':
    L_values = [3, 4, 5, 6, 7, 8]

    results_128 = run_projection(128, L_values)
    results_256 = run_projection(256, L_values)
