"""
Radix-4 validation at N=32: the sternest row of the thesis table.
The b=2 structure needed L=10 there by listening test; radix-4 reaches
full mixing in three layers. Projects the trained matrix onto both
manifolds under the same protocol and reports LSD per configuration.

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

N = 32
GAMMA_LSD = 0.999       #for lsd computation (matches thesis protocol)
GAMMA_RENDER = 0.9999   #for ir rendering (matches training t60)

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
A = np.load(os.path.join(data_dir, f'flamo_trained_N{N}.npy'))
B = np.load(os.path.join(data_dir, f'flamo_trained_N{N}_B.npy')).squeeze()
C = np.load(os.path.join(data_dir, f'flamo_trained_N{N}_C.npy')).squeeze()

#delays recovered from the original training archive
#(riemannian-FDN/archive/experiments/flamo_N32/parameters.mat, whose
#B and C match data/ exactly and whose expm(A - A^T) reproduces the
#trained matrix to float32 precision)
delays = np.load(os.path.join(data_dir, f'flamo_trained_N{N}_delays.npy'))

#re-orthogonalise via polar decomposition (training precision)
U, _, Vt = np.linalg.svd(A)
A = U @ Vt
print(f"N={N}, ||A^T A - I|| = {np.linalg.norm(A.T @ A - np.eye(N)):.2e}")
print(f"delays: {delays[0]}..{delays[-1]}")

out_dir = os.path.join(os.path.dirname(__file__), 'listening_radix4_N32')
os.makedirs(out_dir, exist_ok=True)

#dense reference ir
ir_dense = render_ir(A, delays, gamma=GAMMA_RENDER, b=B, c=C)
sf.write(os.path.join(out_dir, f'N{N}_dense_trained.wav'), ir_dense, 48000)

results = []

#baseline: the thesis row, b=2 at L=10, recomputed under this protocol
t0 = time.perf_counter()
proj = project_dense_to_staged(A, N, L=10, n_restarts=15, max_iter=30)
lsd_frob = log_spectral_distance(A, proj['S'], delays, gamma=GAMMA_LSD,
                                 b=B, c=C, n_freqs=512)
refined = refine_with_lsd(proj, A, delays, gamma=GAMMA_LSD, b=B, c=C,
                          n_freqs=512, maxiter=300, lr=5e-3)
dt = time.perf_counter() - t0
print(f"\nb=2  L=10: frob LSD {lsd_frob:.3f} dB, refined {refined['lsd']:.3f} dB"
      f"  [{dt:.0f}s]")
results.append(('staged b=2 L=10', lsd_frob, refined['lsd'], refined['S']))

#radix-4 at l = 2 (spread 16), 3 (full mixing), 4 (over-complete depth)
for L in [2, 3, 4]:
    t0 = time.perf_counter()
    proj4 = project_dense_to_radix4(A, N, L=L, n_restarts=15, max_iter=30)
    lsd_frob = log_spectral_distance(A, proj4['S'], delays, gamma=GAMMA_LSD,
                                     b=B, c=C, n_freqs=512)
    refined4 = refine_with_lsd_radix4(proj4, A, delays, gamma=GAMMA_LSD,
                                      b=B, c=C, n_freqs=512, maxiter=300,
                                      lr=5e-3)
    dt = time.perf_counter() - t0
    print(f"r4   L={L}:  frob LSD {lsd_frob:.3f} dB, refined {refined4['lsd']:.3f} dB"
          f"  [{dt:.0f}s]")
    results.append((f'radix4 L={L}', lsd_frob, refined4['lsd'], refined4['S']))

#render every configuration for listening
print(f"\n{'config':<18} {'frob lsd':>10} {'refined lsd':>12}")
for name, lf, lr_, S in results:
    print(f"{name:<18} {lf:>10.3f} {lr_:>12.3f}")
    ir = render_ir(S, delays, gamma=GAMMA_RENDER, b=B, c=C)
    sf.write(os.path.join(out_dir, f"N{N}_{name.replace(' ', '_').replace('=', '')}.wav"),
             ir, 48000)

print(f"\nwavs in {out_dir}")
