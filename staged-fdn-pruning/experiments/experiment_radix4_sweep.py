"""
Radix-4 candidate validation across the remaining sizes.
The b=2 layer counts were chosen by listening test; those choices do
not transfer to the b=4 manifold, as the N=32 case showed (LSD parity
at L=3, rejected by ear, accepted at L=4). This projects each trained
matrix onto radix-4 at candidate depths and renders IRs for listening.

Author: Facundo Franchino
"""

import sys, os, time
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from staged_fdn.projection import (
    project_dense_to_staged, refine_with_lsd, log_spectral_distance,
)
from staged_fdn.radix4 import (
    project_dense_to_radix4, refine_with_lsd_radix4,
)
from project_large_N import render_ir

GAMMA_LSD = 0.999
GAMMA_RENDER = 0.9999

#per size: the thesis b=2 layer count and the radix-4 candidates
CONFIGS = {
    16:  {'staged_L': 4,  'r4_L': [2, 3]},
    64:  {'staged_L': 3,  'r4_L': [2, 3]},
}


def run_size(N, staged_L, r4_Ls, precision='double'):
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    A = np.load(os.path.join(data_dir, f'flamo_trained_N{N}.npy'))
    B = np.load(os.path.join(data_dir, f'flamo_trained_N{N}_B.npy')).squeeze()
    C = np.load(os.path.join(data_dir, f'flamo_trained_N{N}_C.npy')).squeeze()
    delays = np.load(os.path.join(data_dir, f'flamo_trained_N{N}_delays.npy'))

    #re-orthogonalise via polar decomposition (training precision)
    U, _, Vt = np.linalg.svd(A)
    A = U @ Vt
    print(f"\nN={N}, delays {delays[0]}..{delays[-1]}, "
          f"||A^T A - I|| = {np.linalg.norm(A.T @ A - np.eye(N)):.2e}")

    out_dir = os.path.join(os.path.dirname(__file__), f'listening_radix4_N{N}')
    os.makedirs(out_dir, exist_ok=True)

    ir = render_ir(A, delays, gamma=GAMMA_RENDER, b=B, c=C)
    sf.write(os.path.join(out_dir, f'N{N}_dense_trained.wav'), ir, 48000)

    results = []

    #b=2 baseline at the listening-chosen depth
    t0 = time.perf_counter()
    proj = project_dense_to_staged(A, N, L=staged_L, n_restarts=15, max_iter=30)
    refined = refine_with_lsd(proj, A, delays, gamma=GAMMA_LSD, b=B, c=C,
                              n_freqs=512, maxiter=300, lr=5e-3,
                              precision=precision)
    print(f"  b=2 L={staged_L}: refined LSD {refined['lsd']:.3f} dB "
          f"[{time.perf_counter()-t0:.0f}s]")
    results.append((f'staged b=2 L={staged_L}', refined['lsd'], refined['S']))

    for L in r4_Ls:
        t0 = time.perf_counter()
        proj4 = project_dense_to_radix4(A, N, L=L, n_restarts=15, max_iter=30)
        refined4 = refine_with_lsd_radix4(proj4, A, delays, gamma=GAMMA_LSD,
                                          b=B, c=C, n_freqs=512, maxiter=300,
                                          lr=5e-3, precision=precision)
        print(f"  r4  L={L}: refined LSD {refined4['lsd']:.3f} dB "
              f"[{time.perf_counter()-t0:.0f}s]")
        results.append((f'radix4 L={L}', refined4['lsd'], refined4['S']))

    #two decay regimes: the long tail is the harder test, the short one
    #matches the original project_large_N listening protocol
    for g, tag in [(GAMMA_RENDER, ''), (0.999, '_g999')]:
        ir = render_ir(A, delays, gamma=g, b=B, c=C)
        sf.write(os.path.join(out_dir, f'N{N}_dense_trained{tag}.wav'), ir, 48000)
        for name, lsd, S in results:
            ir = render_ir(S, delays, gamma=g, b=B, c=C)
            fname = f"N{N}_{name.replace(' ', '_').replace('=', '')}{tag}.wav"
            sf.write(os.path.join(out_dir, fname), ir, 48000)

    print(f"  {'config':<18} {'refined lsd':>12}")
    for name, lsd, _ in results:
        print(f"  {name:<18} {lsd:>12.3f}")
    print(f"  wavs in {out_dir}")
    return results


if __name__ == '__main__':
    for N, cfg in CONFIGS.items():
        run_size(N, cfg['staged_L'], cfg['r4_L'])
