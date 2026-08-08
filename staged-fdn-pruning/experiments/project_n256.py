"""Project N=256 with re-orthogonalised matrix."""
import sys, os
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from staged_fdn.projection import (
    project_dense_to_staged, refine_with_lsd,
    log_spectral_distance
)
from project_large_N import render_ir

A = np.load('flamo_trained_N256.npy')
B = np.load('flamo_trained_N256_B.npy')
C = np.load('flamo_trained_N256_C.npy')
delays = np.load('flamo_trained_N256_delays.npy')
N = 256

#re-orthogonalise via polar decomposition
U, _, Vt = np.linalg.svd(A)
A = U @ Vt
print(f"||A^T A - I|| after polar: {np.linalg.norm(A.T @ A - np.eye(N)):.2e}")
np.save('flamo_trained_N256.npy', A)

gamma = 0.999
out_dir = 'listening_L_comparison'

#dense ir
print("Rendering dense IR...")
ir_dense = render_ir(A, delays, gamma=gamma, b=B, c=C)
sf.write(os.path.join(out_dir, 'N256_dense_trained.wav'), ir_dense, 48000)

for L in [3, 4, 5, 6, 7, 8]:
    print(f"\nL={L}:")
    proj = project_dense_to_staged(A, N, L=L, n_restarts=15, max_iter=30, verbose=False)
    lsd_frob = log_spectral_distance(A, proj['S'], delays, gamma=gamma, b=B, c=C, n_freqs=512)
    print(f"  Frobenius LSD: {lsd_frob:.2f} dB")

    refined = refine_with_lsd(proj, A, delays, gamma=gamma, b=B, c=C,
                               n_freqs=512, maxiter=300, lr=5e-3, verbose=True)
    print(f"  Refined LSD:   {refined['lsd']:.2f} dB")

    ir = render_ir(refined['S'], delays, gamma=gamma, b=B, c=C)
    sf.write(os.path.join(out_dir, f'N256_L{L}_trained.wav'), ir, 48000)
