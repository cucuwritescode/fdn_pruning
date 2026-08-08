"""
Experiment 24: Echo Density & Perceptual Evaluation
====================================================
Evaluates CMA-ES optimised winners on temporal/perceptual metrics:
  - Echo density curve (Abel-Huang measure)
  - Mixing time (time to 0.95 echo density)
  - Spectral flatness over time
  - Listening WAV renders (IR, convolved speech, convolved clicks)

Author: Facundo Franchino
"""

import numpy as np
import json
import os
from scipy.signal import find_peaks, stft
from scipy.linalg import block_diag
import scipy.io.wavfile as wavfile

#============================================================================
#constants
#============================================================================

FS = 48000
T60 = 2.0
IR_DURATION = 2.0
MASTER_SEED = 42

DELAY_SETS = {
    6:  [97, 113, 127, 139, 151, 163],
    8:  [97, 113, 127, 139, 151, 163, 173, 191],
    16: [97, 113, 127, 139, 151, 163, 173, 191, 199, 211, 223, 233, 241, 251, 263, 269],
}


#============================================================================
#matrix builders (from experiment 23)
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

def _perfect_shuffle_perm(N):
    half = N // 2
    perm = np.zeros(N, dtype=int)
    for i in range(half):
        perm[2*i] = i
        perm[2*i+1] = half + i
    if N % 2 == 1:
        perm[N-1] = N-1
    return perm

def _angles_to_ortho_2x2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def build_staged_b2_L3_PS(params, N):
    perm_schedule = [_perfect_shuffle_perm(N) for _ in range(3)]
    n_full = N // 2
    remainder = N % 2
    A = np.eye(N)
    idx = 0
    for l in range(3):
        blocks = []
        for b in range(n_full):
            blocks.append(_angles_to_ortho_2x2(params[idx]))
            idx += 1
        if remainder > 0:
            blocks.append(np.array([[1.0]]))
        B = block_diag(*blocks)
        P = np.eye(N)[perm_schedule[l]]
        A = P @ B @ A
    return A

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


#============================================================================
#fdn ir generation
#============================================================================

def generate_ir(A, delays, condition="lossless", fs=FS, duration=IR_DURATION):
    N = A.shape[0]
    total_samples = int(duration * fs)
    if condition == "lossless":
        gamma = 1.0
        alpha = 0.0
    else:
        gamma = 10 ** (-3.0 / (T60 * fs))
        alpha = 0.2
    gamma_m = gamma ** np.array(delays)
    buffers = [np.zeros(m) for m in delays]
    pointers = np.zeros(N, dtype=int)
    filter_states = np.zeros(N)
    b_vec = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]) / np.sqrt(N)
    c_vec = np.array([1.0 if (i // 2) % 2 == 0 else -1.0 for i in range(N)]) / np.sqrt(N)
    ir = np.zeros(total_samples)
    for n in range(total_samples):
        outs = np.array([buffers[i][pointers[i]] for i in range(N)])
        if alpha > 0:
            filter_states = (1 - alpha) * outs + alpha * filter_states
            filtered = filter_states
        else:
            filtered = outs
        ir[n] = np.dot(c_vec, filtered)
        in_val = 1.0 if n == 0 else 0.0
        feedback = A @ (filtered * gamma_m) + b_vec * in_val
        for i in range(N):
            buffers[i][pointers[i]] = feedback[i]
            pointers[i] = (pointers[i] + 1) % delays[i]
    return ir


#============================================================================
#echo density (abel-huang measure)
#============================================================================

def echo_density_curve(ir, fs=FS, window_ms=10):
    """
    Abel-Huang echo density: fraction of energy samples in each window
    that exceed the expected standard deviation for Gaussian noise.

    In a Gaussian process, ~31.7% of samples exceed 1 std.
    Echo density = 1 when the IR is fully diffuse (Gaussian-like).
    """
    window_samples = int(window_ms * fs / 1000)
    n_windows = len(ir) // window_samples
    times = np.zeros(n_windows)
    densities = np.zeros(n_windows)

    for w in range(n_windows):
        start = w * window_samples
        end = start + window_samples
        chunk = ir[start:end]
        if len(chunk) == 0:
            continue

        #local energy normalisation
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms < 1e-20:
            densities[w] = 0.0
        else:
            #fraction of samples exceeding 1 std (rms)
            exceeding = np.sum(np.abs(chunk) > rms)
            #for gaussian noise, this fraction is ~0.3173
            #normalise so that gaussian = 1.0
            densities[w] = (exceeding / len(chunk)) / 0.3173

        times[w] = (start + window_samples // 2) / fs

    return times, densities


def mixing_time(ir, fs=FS, window_ms=10, threshold=0.95):
    """Time (ms) at which echo density first exceeds threshold."""
    times, densities = echo_density_curve(ir, fs, window_ms)
    for i, d in enumerate(densities):
        if d >= threshold:
            return times[i] * 1000  #convert to ms
    return float('inf')


#============================================================================
#spectral flatness over time
#============================================================================

def spectral_flatness_over_time(ir, fs=FS, window_ms=20, hop_ms=5, n_fft=2048):
    """
    Spectral flatness (Wiener entropy) in sliding windows.
    1.0 = white noise, 0.0 = tonal.
    """
    window_samples = int(window_ms * fs / 1000)
    hop_samples = int(hop_ms * fs / 1000)

    times = []
    flatnesses = []

    pos = 0
    while pos + window_samples <= len(ir):
        chunk = ir[pos:pos + window_samples]
        spectrum = np.abs(np.fft.rfft(chunk, n=n_fft)) ** 2
        spectrum = spectrum[1:]  #drop dc
        spectrum = np.maximum(spectrum, 1e-30)

        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)
        flatness = geo_mean / arith_mean if arith_mean > 0 else 0.0

        times.append((pos + window_samples // 2) / fs * 1000)  #ms
        flatnesses.append(float(flatness))
        pos += hop_samples

    return np.array(times), np.array(flatnesses)


def flatness_at_times(ir, target_ms=[50, 100, 200], fs=FS):
    """Spectral flatness at specific time points."""
    times, flatnesses = spectral_flatness_over_time(ir, fs)
    result = {}
    for t_ms in target_ms:
        idx = np.argmin(np.abs(times - t_ms))
        result[f"{t_ms}ms"] = round(float(flatnesses[idx]), 4) if idx < len(flatnesses) else 0.0
    return result


#============================================================================
#wav writing
#============================================================================

def save_wav(ir, path, fs=FS):
    peak = np.max(np.abs(ir))
    if peak > 0:
        ir_norm = (ir / peak * 0.9 * 32767).astype(np.int16)
    else:
        ir_norm = np.zeros_like(ir, dtype=np.int16)
    wavfile.write(path, fs, ir_norm)


def generate_click_train(fs=FS, duration=1.0, interval_ms=200):
    """Generate a click train for convolution testing."""
    total = int(duration * fs)
    interval = int(interval_ms * fs / 1000)
    signal = np.zeros(total)
    for i in range(0, total, interval):
        signal[i] = 1.0
    return signal


def convolve_and_save(ir, dry_signal, path, fs=FS):
    """Convolve IR with dry signal, normalise, save as WAV."""
    wet = np.convolve(dry_signal, ir)[:len(dry_signal) + len(ir)]
    peak = np.max(np.abs(wet))
    if peak > 0:
        wet_norm = (wet / peak * 0.9 * 32767).astype(np.int16)
    else:
        wet_norm = np.zeros(len(wet), dtype=np.int16)
    wavfile.write(path, fs, wet_norm)


#============================================================================
#main evaluation
#============================================================================

def evaluate_config(name, A, N, delays, output_dir):
    """Full perceptual evaluation of one matrix config."""
    print(f"\n  {name} (N={N})...")
    os.makedirs(output_dir, exist_ok=True)

    result = {"name": name, "N": N}

    #matrix properties
    nnz = int(np.sum(np.abs(A) > 1e-12))
    result["density"] = round(nnz / (N * N), 4)
    result["orth_error"] = float(np.linalg.norm(A.T @ A - np.eye(N)))

    #generate irs
    for cond in ["lossless", "attenuated"]:
        ir = generate_ir(A, delays, condition=cond)

        #echo density
        times_ed, densities_ed = echo_density_curve(ir)
        mt = mixing_time(ir)
        result[f"mixing_time_{cond}_ms"] = round(mt, 1)

        #mean echo density after 50ms
        idx_50ms = np.searchsorted(times_ed, 0.05)
        if idx_50ms < len(densities_ed):
            result[f"echo_density_mean_50ms_{cond}"] = round(float(np.mean(densities_ed[idx_50ms:])), 4)
        else:
            result[f"echo_density_mean_50ms_{cond}"] = 0.0

        #spectral flatness at key times
        sf = flatness_at_times(ir, [50, 100, 200])
        for k, v in sf.items():
            result[f"spectral_flatness_{k}_{cond}"] = v

        #save ir wav
        save_wav(ir, os.path.join(output_dir, f"{name}_ir_{cond}.wav"))

        #save echo density data
        np.savez(os.path.join(output_dir, f"{name}_echo_density_{cond}.npz"),
                 times=times_ed, densities=densities_ed)

        #print key metrics
        print(f"    {cond}: mixing_time={mt:.1f}ms  echo_dens_mean={result[f'echo_density_mean_50ms_{cond}']:.3f}  "
              f"flatness@100ms={sf['100ms']:.4f}")

    #convolved clicks (attenuated only)
    ir_att = generate_ir(A, delays, condition="attenuated")
    clicks = generate_click_train()
    convolve_and_save(ir_att, clicks, os.path.join(output_dir, f"{name}_clicks.wav"))

    return result


def main():
    print("*" * 70)
    print("  EXPERIMENT 24: ECHO DENSITY & PERCEPTUAL EVALUATION")
    print("*" * 70)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "echo_density_eval")
    os.makedirs(output_dir, exist_ok=True)

    #load cma-es results
    cmaes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmaes_output", "cmaes_results.json")
    with open(cmaes_path) as f:
        cmaes = json.load(f)

    all_results = []

    #cmv best at n=6, n=8, n=16
    for Nkey in ["N6", "N8", "N16"]:
        N = int(Nkey[1:])
        delays = DELAY_SETS[N]
        params = np.array(cmaes[Nkey]["cmv"]["params"])
        A = build_cmv(params, N)
        r = evaluate_config(f"cmv_opt_{Nkey}", A, N, delays, output_dir)
        all_results.append(r)

    #staged b=2 l=3 ps best at n=6, n=8
    for Nkey in ["N6", "N8"]:
        N = int(Nkey[1:])
        delays = DELAY_SETS[N]
        params = np.array(cmaes[Nkey]["staged_b2_L3_PS"]["params"])
        A = build_staged_b2_L3_PS(params, N)
        r = evaluate_config(f"staged_b2_PS_opt_{Nkey}", A, N, delays, output_dir)
        all_results.append(r)

    #dense orthogonal control at n=8
    N = 8
    A = gen_dense_orthogonal(N, MASTER_SEED)
    r = evaluate_config("dense_ortho_ctrl_N8", A, N, DELAY_SETS[N], output_dir)
    all_results.append(r)

    #householder k=2 control at n=8
    A = build_k_householder(N, 2, MASTER_SEED)
    r = evaluate_config("householder_K2_ctrl_N8", A, N, DELAY_SETS[N], output_dir)
    all_results.append(r)

    #summary table
    print()
    print("=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(f"  {'Config':30s} {'N':>3s} {'dens':>6s} {'mix_ll':>8s} {'mix_att':>8s} "
          f"{'ed_ll':>7s} {'ed_att':>7s} {'sf@100_ll':>9s} {'sf@100_att':>10s}")
    print("  " + "-" * 95)
    for r in all_results:
        print(f"  {r['name']:30s} {r['N']:3d} {r['density']:6.3f} "
              f"{r.get('mixing_time_lossless_ms', 0):8.1f} {r.get('mixing_time_attenuated_ms', 0):8.1f} "
              f"{r.get('echo_density_mean_50ms_lossless', 0):7.3f} {r.get('echo_density_mean_50ms_attenuated', 0):7.3f} "
              f"{r.get('spectral_flatness_100ms_lossless', 0):9.4f} {r.get('spectral_flatness_100ms_attenuated', 0):10.4f}")

    #save
    json_path = os.path.join(output_dir, "echo_density_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results: {json_path}")
    print(f"  WAVs: {output_dir}/")


if __name__ == "__main__":
    main()
