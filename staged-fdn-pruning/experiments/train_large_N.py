"""
Train dense FDN at N=128 and N=256 using DiffFDN.
Same protocol as existing trained models (N=4 to 64).

Author: Facundo Franchino, 2026
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'diff-fdn-colorless'))

from sympy import nextprime
from model import DiffFDN
from trainer import Trainer
from dataloader import load_dataset
from utility import save_parametes, weights_init_normal


def jot_delays(N, low_ms=12, high_ms=35, fs=48000):
    """Generate N coprime delay lengths in a perceptually useful range.

    Jot-distributed: logarithmically spaced, then rounded to nearest prime
    for coprimality. Matches the protocol used for N=4 through 64.
    """
    low_samp = int(low_ms * fs / 1000)
    high_samp = int(high_ms * fs / 1000)
    #logarithmic spacing
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


def extract_orthogonal_A(net):
    """Extract the dense orthogonal feedback matrix from a trained DiffFDN."""
    with torch.no_grad():
        A_raw = net.A.data
        A_ortho = net.ortho_param(A_raw)
    return A_ortho.cpu().numpy()


def train_and_save(N, device='mps', max_epochs=10, lr=1e-3, alpha=2,
                   gain_per_sample=0.9999, batch_size=2000, num_samples=None):
    delays = jot_delays(N)
    print(f"\n{'='*60}")
    print(f"Training N={N}, delays range [{delays[0]}, {delays[-1]}]")
    print(f"Device: {device}, epochs: {max_epochs}, lr: {lr}, alpha: {alpha}")
    print(f"{'='*60}\n")

    #build args namespace matching solver.py expectations
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
    args.num = num_samples if num_samples else 480000
    args.split = 0.8
    args.shuffle = True
    args.train_dir = os.path.join(os.path.dirname(__file__), f'trained_N{N}')
    os.makedirs(args.train_dir, exist_ok=True)

    #dataset
    train_dataset, valid_dataset = load_dataset(args)

    #model
    net = DiffFDN(args.delays, args.gain_per_sample, args.device,
                  args.scattering, args.householder,
                  block_diagonal=args.block_diagonal)
    net.apply(weights_init_normal)

    #move everything to device before Trainer init (which calls normalize then forward)
    net.to(device)
    net.m = net.m.to(device)

    #train
    trainer = Trainer(net, args)
    trainer.train(train_dataset, valid_dataset)

    #extract and save
    A = extract_orthogonal_A(trainer.net)
    B = trainer.net.B.data.cpu().numpy().squeeze()
    C = trainer.net.C.data.cpu().numpy().squeeze()

    out_dir = os.path.dirname(__file__)
    np.save(os.path.join(out_dir, f'flamo_trained_N{N}.npy'), A)
    np.save(os.path.join(out_dir, f'flamo_trained_N{N}_B.npy'), B)
    np.save(os.path.join(out_dir, f'flamo_trained_N{N}_C.npy'), C)
    np.save(os.path.join(out_dir, f'flamo_trained_N{N}_delays.npy'), np.array(delays))

    #verify orthogonality
    err = np.linalg.norm(A.T @ A - np.eye(N))
    print(f"\nSaved flamo_trained_N{N}.npy")
    print(f"  A shape: {A.shape}")
    print(f"  ||A^T A - I|| = {err:.2e}")
    print(f"  Delays: [{delays[0]}, ..., {delays[-1]}]")

    return A


if __name__ == '__main__':
    #cpu only: matrix_exp not on MPS, and Trainer's frequency
    #tensors don't get moved to device.
    #reduced dataset size for large N (matrix_exp is O(N^3) per batch).
    train_and_save(128, device='cpu', batch_size=2000, num_samples=20000)
    train_and_save(256, device='cpu', batch_size=2000, num_samples=20000)
