"""
Train dense FDN at N=128 and N=256, then project at L=3..8 and render IRs.
Matches the exact protocol used for N=4 through N=64.

Author: Facundo Franchino, 2026
"""

import sys, os
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'diff-fdn-colorless'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import nextprime
from staged_fdn.projection import (
    project_dense_to_staged, refine_with_lsd,
    log_spectral_distance
)

#============================================================================
#delay generation, same as N=4..64
#============================================================================

def jot_delays(N, low_ms=12, high_ms=35, fs=48000):
    low_samp = int(low_ms * fs / 1000)
    high_samp = int(high_ms * fs / 1000)
    targets = np.logspace(np.log10(low_samp), np.log10(high_samp), N)
    delays = []
    used = set()
    for t in targets:
        p = int(nextprime(int(t)))
        while p in used:
            p = int(nextprime(p))
        used.add(p)
        delays.append(p)
    return delays

#============================================================================
#ir rendering, time-domain fdn
#============================================================================

def render_ir(A, delays, gamma=0.9999, b=None, c=None, duration_s=2.0, fs=48000):
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
        ir[n] = np.dot(c, state)

        #feedback through the attenuated state, impulse at n=0
        fb = A @ (state * gamma_m)
        if n == 0:
            fb = fb + b

        #write back at the read positions, then advance every pointer
        buf[idx] = fb
        ptrs += 1
        np.mod(ptrs, delays, out=ptrs)
        np.add(offsets, ptrs, out=idx)

    peak = np.max(np.abs(ir))
    if peak > 0:
        ir = ir / peak * 0.9
    return ir

#============================================================================
#training
#============================================================================

def train_dense(N, delays, device='cpu', max_epochs=10, lr=1e-3, alpha=2,
                gain_per_sample=0.9999, batch_size=2000, num_samples=20000):
    import torch
    from model import DiffFDN
    from trainer import Trainer
    from dataloader import load_dataset
    from utility import weights_init_normal

    print(f"\n  Training N={N}, delays {delays[0]/48:.1f}-{delays[-1]/48:.1f} ms")

    class Args:
        pass
    args = Args()
    args.delays = delays
    args.gain_per_sample = gain_per_sample
    args.device = device
    args.scattering = False
    args.householder = False
    args.block_diagonal = False
    args.samplerate = 48000
    args.batch_size = batch_size
    args.max_epochs = max_epochs
    args.lr = lr
    args.alpha = alpha
    args.num = num_samples
    args.split = 0.8
    args.shuffle = True
    args.train_dir = os.path.join(os.path.dirname(__file__), f'trained_N{N}')
    os.makedirs(args.train_dir, exist_ok=True)

    train_dataset, valid_dataset = load_dataset(args)

    net = DiffFDN(args.delays, args.gain_per_sample, args.device,
                  args.scattering, args.householder,
                  block_diagonal=args.block_diagonal)
    net.apply(weights_init_normal)
    net.to(device)
    net.m = net.m.to(device)

    trainer = Trainer(net, args)
    trainer.train(train_dataset, valid_dataset)

    #extract orthogonal A
    with torch.no_grad():
        A = net.ortho_param(net.A.data).cpu().numpy()

    #re-orthogonalise via polar decomposition (training precision)
    U, _, Vt = np.linalg.svd(A)
    A = U @ Vt

    B = net.B.data.cpu().numpy().squeeze()
    C = net.C.data.cpu().numpy().squeeze()

    err = np.linalg.norm(A.T @ A - np.eye(N))
    print(f"  ||A^T A - I|| = {err:.2e}")
    return A, B, C

#============================================================================
#projection and rendering
#============================================================================

def project_and_render(N, A, B, C, delays, L_values):
    gamma_render = 0.9999   #for ir rendering (matches training t60)
    gamma_lsd = 0.999       #for lsd computation (matches thesis protocol)
    out_dir = os.path.join(os.path.dirname(__file__), 'listening_L_comparison')

    #dense reference ir
    print(f"  Rendering dense IR...")
    ir_dense = render_ir(A, delays, gamma=gamma_render, b=B, c=C)
    sf.write(os.path.join(out_dir, f'N{N}_dense_trained.wav'), ir_dense, 48000)

    results = []
    for L in L_values:
        print(f"\n  L={L}:")
        proj = project_dense_to_staged(A, N, L=L, n_restarts=15,
                                        max_iter=30, verbose=False)
        lsd_frob = log_spectral_distance(A, proj['S'], delays, gamma=gamma_lsd,
                                          b=B, c=C, n_freqs=512)
        print(f"    Frobenius LSD: {lsd_frob:.2f} dB")

        refined = refine_with_lsd(proj, A, delays, gamma=gamma_lsd, b=B, c=C,
                                   n_freqs=512, maxiter=300, lr=5e-3, verbose=True)
        print(f"    Refined LSD:   {refined['lsd']:.2f} dB")

        ir_staged = render_ir(refined['S'], delays, gamma=gamma_render, b=B, c=C)
        sf.write(os.path.join(out_dir, f'N{N}_L{L}_trained.wav'), ir_staged, 48000)
        results.append({'L': L, 'lsd_frob': lsd_frob, 'lsd_refined': refined['lsd']})

    print(f"\n  --- N={N} Summary ---")
    print(f"  {'L':>3}  {'LSD Frob':>10}  {'LSD Refined':>12}")
    for r in results:
        print(f"  {r['L']:>3}  {r['lsd_frob']:>10.2f}  {r['lsd_refined']:>12.2f}")
    return results

#============================================================================
#main
#============================================================================

if __name__ == '__main__':
    exp_dir = os.path.dirname(__file__)
    out_dir = os.path.join(exp_dir, 'listening_L_comparison')
    os.makedirs(out_dir, exist_ok=True)

    L_values = [3, 4, 5, 6, 7, 8]

    for N in [128, 256]:
        delays = jot_delays(N)
        print(f"\n{'='*60}")
        print(f"N={N}, delays: {delays[0]/48:.1f}-{delays[-1]/48:.1f} ms")
        print(f"{'='*60}")

        #train
        A, B, C = train_dense(N, delays, device='cpu', num_samples=20000)

        #save
        np.save(os.path.join(exp_dir, f'flamo_trained_N{N}.npy'), A)
        np.save(os.path.join(exp_dir, f'flamo_trained_N{N}_B.npy'), B)
        np.save(os.path.join(exp_dir, f'flamo_trained_N{N}_C.npy'), C)
        np.save(os.path.join(exp_dir, f'flamo_trained_N{N}_delays.npy'), np.array(delays))

        #project and render
        project_and_render(N, A, B, C, delays, L_values)
