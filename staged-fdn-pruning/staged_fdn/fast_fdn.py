"""
optimised IR generation for feedback delay networks
===============================
One flat buffer holds every delay line; reads and writes are gathered,
so the per-sample cost is a handful of vector operations rather than
a python loop over lines.

Author: Facundo Franchino
"""

import numpy as np

def generate_reverb_ir_fast(A, delays, t60_low, alpha=0.2, fs=48000, duration_seconds=1.0):
    N = A.shape[0]
    total_samples = int(duration_seconds * fs)
    gamma = 10**(-3 / (t60_low * fs))
    delays = np.asarray(delays, dtype=int)
    gamma_m = gamma ** delays

    #state: one flat ring buffer for all delay lines
    #idx marks each line's read head, offset plus pointer
    offsets = np.concatenate([[0], np.cumsum(delays[:-1])])
    buf = np.zeros(int(delays.sum()))
    pointers = np.zeros(N, dtype=int)
    idx = offsets + pointers
    filter_states = np.zeros(N)

    b_vec = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]) / np.sqrt(N)
    c_vec = np.array([1.0 if (i // 2) % 2 == 0 else -1.0 for i in range(N)]) / np.sqrt(N)

    ir = np.zeros(total_samples)
    A = A.astype(np.float64)

    for n in range(total_samples):
        #read every delay line head in one gather
        outs = buf[idx]

        #low-pass filter
        filter_states = (1 - alpha) * outs + alpha * filter_states
        ir[n] = np.dot(c_vec, filter_states)

        #input
        in_val = 1.0 if n == 0 else 0.0

        #feedback
        feedback_in = A @ (filter_states * gamma_m) + b_vec * in_val

        #write back at the read positions, then advance every pointer
        buf[idx] = feedback_in
        pointers += 1
        np.mod(pointers, delays, out=pointers)
        np.add(offsets, pointers, out=idx)

    return ir
