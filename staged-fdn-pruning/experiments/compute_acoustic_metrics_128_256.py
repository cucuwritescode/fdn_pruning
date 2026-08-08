"""
Compute sigma_rho and t_mix for trained N=128 and N=256 staged projections.
Uses short proxy delays for modal_std (same as benchmark protocol).

Author: Facundo Franchino, 2026
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'archive', 'experiments'))

from staged_fdn.projection import project_dense_to_staged, refine_with_lsd, build_staged_from_angles, perm_matrix
from experiment_02_structured_validation import modal_std
from experiment_24_echo_density import generate_ir, mixing_time

#short proxy delays for modal_std (same range as benchmark protocol)
def short_proxy_delays(N):
    """Short coprime delays for tractable eigendecomposition.
    Range 7-67, logarithmically spaced, rounded to primes."""
    from sympy import nextprime
    targets = np.logspace(np.log10(7), np.log10(67), N)
    delays = []
    used = set()
    for t in targets:
        p = int(nextprime(int(t)))
        while p in used:
            p = int(nextprime(p))
        used.add(p)
        delays.append(p)
    return delays

GAMMA = 0.9999

for N in [128, 256]:
    print(f"\n{'='*50}")
    print(f"N={N}")
    print(f"{'='*50}")

    A_dense = np.load(f'flamo_trained_N{N}.npy')
    delays_real = np.load(f'flamo_trained_N{N}_delays.npy')
    B = np.load(f'flamo_trained_N{N}_B.npy')
    C = np.load(f'flamo_trained_N{N}_C.npy')

    #re-orthogonalise if needed
    U, _, Vt = np.linalg.svd(A_dense)
    A_dense = U @ Vt

    #project at L=3
    L = 3
    proj = project_dense_to_staged(A_dense, N, L=L, n_restarts=15, max_iter=30, verbose=False)
    refined = refine_with_lsd(proj, A_dense, delays_real, gamma=0.999, b=B, c=C,
                               n_freqs=512, maxiter=300, lr=5e-3, verbose=False)
    S = refined['S']

    #modal_std with short proxy delays (M stays tractable)
    proxy_delays = short_proxy_delays(N)
    M_total = sum(proxy_delays)
    print(f"  Proxy delays: {proxy_delays[0]}-{proxy_delays[-1]}, M={M_total}")

    if M_total > 5000:
        print(f"  M={M_total} too large for eigendecomposition, skipping modal_std")
        sigma_dense = None
        sigma_staged = None
    else:
        sigma_dense, _, _ = modal_std(A_dense, proxy_delays, GAMMA)
        sigma_staged, _, _ = modal_std(S, proxy_delays, GAMMA)
        print(f"  sigma_rho dense:  {sigma_dense:.1f} dB")
        print(f"  sigma_rho staged: {sigma_staged:.1f} dB")

    #mixing time with real delays (time-domain ir, no eigendecomposition)
    ir_dense = generate_ir(A_dense, list(delays_real.astype(int)), condition="lossless")
    ir_staged = generate_ir(S, list(delays_real.astype(int)), condition="lossless")
    mt_dense = mixing_time(ir_dense)
    mt_staged = mixing_time(ir_staged)
    print(f"  t_mix dense:  {mt_dense:.0f} ms")
    print(f"  t_mix staged: {mt_staged:.0f} ms")
