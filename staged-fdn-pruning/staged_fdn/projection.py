"""
Staged Orthogonal Projection: Dense O(N²) to Staged b=2 PS O(N)
================================================================

projects a dense orthogonal matrix onto the manifold of staged orthogonal
matrices with 2×2 rotation blocks and perfect-shuffle permutations:

    S(θ) = P_{L-1} B_{L-1} ··· P_1 B_1 P_0 B_0

Method (two-phase hybrid)
-------------------------
**first phase : alternating closed-form block coordinate ascent**
    Since ||A - S||²_F = 2N - 2 tr(A^T S), we maximise tr(A^T S).
    For each layer l with other layers fixed, define
        G_l = Right_l @ A^T @ Left_l @ P_l
    and the per-block optimal angle is
        θ_k^(l) = atan2(G[2k,2k+1] - G[2k+1,2k], G[2k,2k] + G[2k+1,2k+1]).
    Fast but can get stuck in local optima due to cross-layer coupling.

**second phase: L-BFGS Refinement**
    All 3N/2 angles optimised jointly with analytical gradient.
    Handles cross-layer coupling that alternating misses.

Author: Facundo Franchino
"""

import numpy as np
from scipy.optimize import minimize


#============================================================================
#fdn frequency response & log-spectral distance
#============================================================================

def fdn_frequency_response(A, delays, gamma=1.0, b=None, c=None, n_freqs=2048):
    """Compute FDN frequency response H(e^{jω}) at n_freqs points in [0, π).

    H(z) = c^T D(z) (I - A Γ D(z))^{-1} b

    where D(z) = diag(z^{-m_i}), Γ = diag(γ^{m_i}).
    Fully vectorised: single batched np.linalg.solve over all frequencies.

    Parameters
    ----------
    A : (N, N) feedback matrix
    delays : length-N array of delay line lengths
    gamma : per-sample decay (< 1 for stable response)
    b, c : input/output vectors (default: alternating ±1/√N)
    n_freqs : number of frequency bins in [0, π)

    Returns
    -------
    omega : (n_freqs,) frequency array
    H : (n_freqs,) complex frequency response
    """
    N = A.shape[0]
    delays = np.asarray(delays, dtype=float)
    #default i/o vectors if caller did not provide any
    if b is None:
        b = np.array([(-1.0)**i for i in range(N)]) / np.sqrt(N)
    if c is None:
        c = np.array([(-1.0)**(i // 2) for i in range(N)]) / np.sqrt(N)

    #per-line decay factor, raised to the delay length
    gamma_m = gamma ** delays
    #evaluate the transfer function on a grid of angular frequencies in [0, pi)
    omega = np.linspace(0, np.pi, n_freqs, endpoint=False)

    #phases[k, j] is e^(-j omega_k m_j), one column per delay line
    phases = np.exp(-1j * np.outer(omega, delays))
    #combine decay and phase per delay line, broadcast across frequencies
    col_mult = gamma_m[np.newaxis, :] * phases
    #build the matrix (i minus a gamma d(z)) for every frequency in one go
    M = np.eye(N)[np.newaxis, :, :] - A[np.newaxis, :, :] * col_mult[:, np.newaxis, :]
    #batched solve for the inverse times b at every frequency.
    #b is passed as an explicit stack of column vectors (n_freqs, N, 1); this is
    #the same solve, written to be unambiguous under NumPy >= 2.0's solve rules.
    p = np.linalg.solve(M, np.broadcast_to(b, (n_freqs, N))[..., np.newaxis])[..., 0]
    #project onto c and apply the delay phases to get the scalar response
    H = np.sum(c[np.newaxis, :] * phases * p, axis=1)

    return omega, H


def log_spectral_distance(A1, A2, delays, gamma=1.0, b=None, c=None,
                          n_freqs=2048):
    """Log-spectral distance between two FDN matrices (dB).

    LSD = sqrt( mean_omega ( 20 log10|H1(omega)| - 20 log10|H2(omega)| )^2 )

    Thresholds: <1 dB transparent, <3 dB good, >5 dB audible.
    """
    #frequency responses of the two matrices using the same delays and gains
    _, H1 = fdn_frequency_response(A1, delays, gamma, b, c, n_freqs)
    _, H2 = fdn_frequency_response(A2, delays, gamma, b, c, n_freqs)

    #convert magnitudes to decibels, small epsilon to avoid log of zero
    eps = 1e-30
    H1_db = 20 * np.log10(np.abs(H1) + eps)
    H2_db = 20 * np.log10(np.abs(H2) + eps)

    #root-mean-square of the dB difference across all frequencies
    return float(np.sqrt(np.mean((H1_db - H2_db) ** 2)))


def _freq_response_fast(S, delays_arr, gamma_m, b, c, phases):
    """Vectorised H(ω) with precomputed phases. Returns (K,) complex array."""
    K, N = phases.shape
    col_mult = gamma_m[np.newaxis, :] * phases              #(k, n)
    M = (np.eye(N)[np.newaxis, :, :]
         - S[np.newaxis, :, :] * col_mult[:, np.newaxis, :])  #(k, n, n)
    p = np.linalg.solve(M, np.broadcast_to(b, (K, N))[..., np.newaxis])[..., 0]  #(k, n)
    return np.sum(c[np.newaxis, :] * phases * p, axis=1)     #(k,)


def _lsd_cost_fn(angles_flat, delays_arr, gamma_m, b, c,
                 P_mats, N, L, phases, H_ref_db):
    """LSD cost for scipy minimisation (scalar output, no gradient)."""
    angles = angles_flat.reshape(L, N // 2)
    S = build_staged_from_angles(angles, P_mats, N, L)
    H_s = _freq_response_fast(S, delays_arr, gamma_m, b, c, phases)
    H_s_db = 20 * np.log10(np.abs(H_s) + 1e-30)
    return float(np.sqrt(np.mean((H_ref_db - H_s_db) ** 2)))


def refine_with_lsd(result, A_dense, delays, gamma=1.0, b=None, c=None,
                    n_freqs=512, maxiter=200, lr=1e-2, verbose=False,
                    precision='double'):
    """Refine projection angles to minimise log-spectral distance.

    Uses PyTorch autograd for exact gradients, orders of magnitude faster
    than finite-difference scipy at large N.

    Parameters
    ----------
    result : dict from project_dense_to_staged
    A_dense : (N, N) original dense orthogonal matrix
    delays : length-N delay line lengths
    gamma : per-sample decay (default 1.0 = lossless; use < 1 for finite H)
    b, c : input/output vectors
    n_freqs : frequency resolution for optimisation
    maxiter : optimisation steps
    lr : Adam learning rate
    verbose : print progress
    precision : 'double' (default) or 'single'; single runs the batched
        solve in complex64, roughly 3x faster at N=256, and the final
        metrics are recomputed at double precision either way

    Returns
    -------
    refined : dict with same keys as project_dense_to_staged plus 'lsd',
              'lsd_init'
    """
    import torch

    N = A_dense.shape[0]
    L = result['angles'].shape[0]
    n_blocks = N // 2
    delays_arr = np.asarray(delays, dtype=float)

    if b is None:
        b = np.array([(-1.0)**i for i in range(N)]) / np.sqrt(N)
    if c is None:
        c = np.array([(-1.0)**(i // 2) for i in range(N)]) / np.sqrt(N)

    perm_list = result['perms']
    gamma_m = gamma ** delays_arr

    #working dtypes; the batched solve dominates the step, so single
    #precision buys the speedup while the loss stays well conditioned
    cdtype = torch.complex64 if precision == 'single' else torch.complex128
    rdtype = torch.float32 if precision == 'single' else torch.float64

    #precompute the frequency grid and the phase matrix once, as torch tensors
    omega = np.linspace(0, np.pi, n_freqs, endpoint=False)
    phases_np = np.exp(-1j * np.outer(omega, delays_arr))

    #constants that do not change during optimisation
    phases_t = torch.tensor(phases_np, dtype=cdtype)
    gamma_m_t = torch.tensor(gamma_m, dtype=rdtype)
    b_t = torch.tensor(b, dtype=cdtype)
    c_t = torch.tensor(c, dtype=cdtype)
    eye_t = torch.eye(N, dtype=cdtype).unsqueeze(0)
    perm_indices = [torch.tensor(p, dtype=torch.long) for p in perm_list]

    #compute the dense reference frequency response once, with no gradient
    A_dense_t = torch.tensor(A_dense, dtype=cdtype)
    col_mult_ref = gamma_m_t.unsqueeze(0).to(cdtype) * phases_t
    M_ref = eye_t - A_dense_t.unsqueeze(0) * col_mult_ref.unsqueeze(1)
    p_ref = torch.linalg.solve(M_ref, b_t.expand(n_freqs, N))
    H_ref = torch.sum(c_t.unsqueeze(0) * phases_t * p_ref, dim=1)
    H_ref_db = 20.0 * torch.log10(torch.abs(H_ref) + 1e-30)
    H_ref_db = H_ref_db.detach()

    #the angles are the trainable parameters, initialised from phase 1
    angles_param = torch.nn.Parameter(
        torch.tensor(result['angles'].ravel(), dtype=rdtype)
    )
    optimizer = torch.optim.Adam([angles_param], lr=lr)

    def forward(angles_flat):
        #build s = p_{l-1} b_{l-1} ... p_0 b_0 from the current angles.
        #each layer is a batched 2x2 multiply on consecutive row pairs,
        #then a row permutation; no python loop over blocks
        angles = angles_flat.reshape(L, n_blocks)
        cv, sv = torch.cos(angles), torch.sin(angles)
        rot = torch.stack([torch.stack([cv, -sv], dim=-1),
                           torch.stack([sv, cv], dim=-1)], dim=-2).to(cdtype)
        S = torch.eye(N, dtype=cdtype)
        for l in range(L):
            S = torch.bmm(rot[l], S.reshape(n_blocks, 2, N)).reshape(N, N)
            S = S[perm_indices[l]]
        #compute the staged frequency response the same way as the dense one
        col_mult = gamma_m_t.unsqueeze(0).to(cdtype) * phases_t
        M = eye_t - S.unsqueeze(0) * col_mult.unsqueeze(1)
        p = torch.linalg.solve(M, b_t.expand(n_freqs, N))
        H = torch.sum(c_t.unsqueeze(0) * phases_t * p, dim=1)
        H_db = 20.0 * torch.log10(torch.abs(H) + 1e-30)
        #log-spectral distance between the staged response and the dense reference
        lsd = torch.sqrt(torch.mean((H_ref_db - H_db) ** 2))
        return lsd, S

    #record the initial lsd before optimisation begins
    with torch.no_grad():
        lsd_init_val, _ = forward(angles_param)
    lsd_init = float(lsd_init_val)
    if verbose:
        print(f"  LSD before refinement: {lsd_init:.2f} dB")

    #gradient descent loop: autograd computes the gradient of lsd w.r.t. all angles
    best_lsd = lsd_init
    best_angles = angles_param.data.clone()
    for step in range(maxiter):
        optimizer.zero_grad()
        lsd, _ = forward(angles_param)
        #backpropagate the lsd through the whole computation graph
        lsd_val = float(lsd.detach())
        #The loss describes the current angles, before the Adam update.
        if lsd_val < best_lsd:
            best_lsd = lsd_val
            best_angles = angles_param.detach().clone()
        lsd.backward()
        optimizer.step()
        if verbose and (step + 1) % 50 == 0:
            print(f"    step {step+1}/{maxiter}: LSD = {lsd_val:.4f} dB")

    #The last update has not yet been scored by the loop.
    with torch.no_grad():
        final_lsd, _ = forward(angles_param)
        if float(final_lsd) < best_lsd:
            best_angles = angles_param.detach().clone()
    #build final result using best angles
    angles_np = best_angles.numpy().reshape(L, n_blocks)
    P_mats = [perm_matrix(p, N) for p in perm_list]
    S_refined = build_staged_from_angles(angles_np, P_mats, N, L)
    frob_error = np.linalg.norm(A_dense - S_refined, 'fro')

    #final lsd at numpy precision
    perm_list_out = perm_list
    gamma_m_np = gamma ** delays_arr
    phases_final = np.exp(-1j * np.outer(omega, delays_arr))
    H_ref_np = _freq_response_fast(A_dense, delays_arr, gamma_m_np, b, c, phases_final)
    H_s_np = _freq_response_fast(S_refined, delays_arr, gamma_m_np, b, c, phases_final)
    H_ref_db_np = 20 * np.log10(np.abs(H_ref_np) + 1e-30)
    H_s_db_np = 20 * np.log10(np.abs(H_s_np) + 1e-30)
    lsd_final = float(np.sqrt(np.mean((H_ref_db_np - H_s_db_np) ** 2)))

    if verbose:
        print(f"  LSD after refinement:  {lsd_final:.2f} dB "
              f"(Δ = {lsd_final - lsd_init:+.2f})")
        print(f"  Frobenius error:       {frob_error:.3f}")

    return {
        'angles': angles_np,
        'perms': perm_list_out,
        'S': S_refined,
        'lsd': lsd_final,
        'lsd_init': lsd_init,
        'frob_error': frob_error,
        'cost': float(np.trace(A_dense.T @ S_refined)),
        'rel_trace': float(np.trace(A_dense.T @ S_refined)) / N,
    }


#============================================================================
#permutation helpers
#============================================================================

def perfect_shuffle_perm(N):
    """Perfect shuffle: interleave first and second halves."""
    half = N // 2
    perm = np.zeros(N, dtype=int)
    for i in range(half):
        perm[2 * i] = i
        perm[2 * i + 1] = half + i
    if N % 2 == 1:
        perm[N - 1] = N - 1
    return perm


def bit_reversal_perm(N):
    """Bit-reversal permutation (works best for powers of 2)."""
    bits = max(1, int(np.ceil(np.log2(N)))) if N > 1 else 1
    perm = np.zeros(N, dtype=int)
    for i in range(N):
        rev = int(format(i, f'0{bits}b')[::-1], 2)
        perm[i] = rev % N
    seen = set()
    out = []
    for p in perm:
        while p in seen:
            p = (p + 1) % N
        seen.add(p)
        out.append(p)
    return np.array(out, dtype=int)


def prime_stride_perm(N, stride=None):
    """Stride permutation with a prime stride coprime to N."""
    if stride is None:
        #pick a prime roughly n/3 to maximise mixing distance
        for s in [max(3, N // 3), max(5, N // 3 + 2), 7, 11, 13]:
            if N % s != 0:
                stride = s
                break
        else:
            stride = 3
    perm = np.array([(i * stride) % N for i in range(N)], dtype=int)
    seen = set()
    out = []
    for p in perm:
        while p in seen:
            p = (p + 1) % N
        seen.add(p)
        out.append(p)
    return np.array(out, dtype=int)


def heterogeneous_perms(N, L):
    """Generate L different permutations for maximum coupling coverage.

    Layer schedule: perfect-shuffle, bit-reversal, prime-stride, then cycle.
    Each permutation couples different index pairs → expander-like mixing.
    Cost: zero arithmetic (just index remapping).
    """
    generators = [
        lambda: perfect_shuffle_perm(N),
        lambda: bit_reversal_perm(N),
        lambda: prime_stride_perm(N),
        lambda: prime_stride_perm(N, stride=max(5, N // 5)),
    ]
    return [generators[l % len(generators)]() for l in range(L)]


def perm_matrix(perm, N):
    """Permutation vector → N×N permutation matrix."""
    return np.eye(N)[perm]


#============================================================================
#building staged matrices from angles
#============================================================================

def build_block_diag_rotations(angles, N):
    """Block-diagonal of 2×2 rotations. angles: (N//2,) array."""
    n_blocks = N // 2
    B = np.eye(N)
    for k in range(n_blocks):
        c, s = np.cos(angles[k]), np.sin(angles[k])
        i = 2 * k
        B[i, i] = c;       B[i, i + 1] = -s
        B[i + 1, i] = s;   B[i + 1, i + 1] = c
    return B


def build_staged_from_angles(angles, P_mats, N, L):
    """Build S = P_{L-1} B_{L-1} ··· P_0 B_0 from (L, N//2) angle array."""
    S = np.eye(N)
    for l in range(L):
        B = build_block_diag_rotations(angles[l], N)
        S = P_mats[l] @ B @ S
    return S


def apply_staged(x, angles, perms, N, L):
    """O(N·L) matrix-vector multiply without forming the dense matrix.

    perms: list of L permutation index arrays (not matrices).
    """
    y = x.copy().astype(float)
    n_blocks = N // 2
    for l in range(L):
        for k in range(n_blocks):
            i = 2 * k
            c, s = np.cos(angles[l, k]), np.sin(angles[l, k])
            a, b = y[i], y[i + 1]
            y[i] = c * a - s * b
            y[i + 1] = s * a + c * b
        y = y[perms[l]]
    return y


#============================================================================
#partial products (shared by alternating and gradient computation)
#============================================================================

def _compute_partial_products(angles, P_mats, N, L, target_layer):
    """Compute Right_l and Left_l for the target layer.

    Right_l = Φ_{l-1} ··· Φ_0   (forward product of layers below l)
    Left_l  = Φ_{L-1} ··· Φ_{l+1}  (backward product of layers above l)
    """
    Right = np.eye(N)
    for j in range(target_layer):
        B = build_block_diag_rotations(angles[j], N)
        Right = P_mats[j] @ B @ Right

    #accumulate from l+1 upward: left = Φ[l-1] @ ... @ Φ[l+1]
    Left = np.eye(N)
    for j in range(target_layer + 1, L):
        B = build_block_diag_rotations(angles[j], N)
        Left = P_mats[j] @ B @ Left

    return Right, Left


#============================================================================
#phase 1: greedy init + alternating closed-form
#============================================================================

def _greedy_init(A, P_mats, N, L):
    """Greedy outside-in layer peeling."""
    n_blocks = N // 2
    angles = np.zeros((L, n_blocks))
    residual = A.copy()

    for l in range(L - 1, -1, -1):
        M = P_mats[l].T @ residual
        for k in range(n_blocks):
            i = 2 * k
            a = M[i, i] + M[i + 1, i + 1]
            b = M[i + 1, i] - M[i, i + 1]
            angles[l, k] = np.arctan2(b, a)
        B = build_block_diag_rotations(angles[l], N)
        residual = B.T @ P_mats[l].T @ residual

    return angles


def _alternating_optimize(A, angles, P_mats, N, L, max_iter, tol):
    """Alternating closed-form block coordinate ascent (fast warm-start).

    The atan2 theorem says: for any single rotation angle, with all other
    angles held fixed, the value that minimises the frobenius distance
    between the staged matrix and the dense target a has a closed-form.
    we sweep through all angles, applying the formula, until the trace
    objective stops improving.
    """
    n_blocks = N // 2
    #build the current staged matrix and record the starting trace value
    S = build_staged_from_angles(angles, P_mats, N, L)
    old_cost = np.trace(A.T @ S)

    for iteration in range(max_iter):
        #forward sweep: update each layer in turn from l = 0 to l-1
        for l in range(L):
            #form the helper matrix g that absorbs everything outside layer l
            Right, Left = _compute_partial_products(angles, P_mats, N, L, l)
            G = Right @ A.T @ Left @ P_mats[l]
            #for each 2x2 block in this layer, read four entries and call atan2
            for k in range(n_blocks):
                i = 2 * k
                a = G[i, i] + G[i + 1, i + 1]
                b = G[i, i + 1] - G[i + 1, i]
                angles[l, k] = np.arctan2(b, a)

        #backward sweep over the layers helps escape plateaus
        for l in range(L - 1, -1, -1):
            Right, Left = _compute_partial_products(angles, P_mats, N, L, l)
            G = Right @ A.T @ Left @ P_mats[l]
            for k in range(n_blocks):
                i = 2 * k
                a = G[i, i] + G[i + 1, i + 1]
                b = G[i, i + 1] - G[i + 1, i]
                angles[l, k] = np.arctan2(b, a)

        #check whether this full sweep improved the trace meaningfully
        S = build_staged_from_angles(angles, P_mats, N, L)
        new_cost = np.trace(A.T @ S)
        if new_cost - old_cost < tol:
            return angles, new_cost
        old_cost = new_cost

    return angles, old_cost


#============================================================================
#phase 2: l-bfgs with analytical gradient
#============================================================================

def _neg_cost_and_grad(angles_flat, A, P_mats, N, L):
    """Compute -tr(A^T S) and its gradient (for scipy minimisation).

    Gradient per block k of layer l:
        ∂tr/∂θ = (G[i,i+1] - G[i+1,i]) cos θ - (G[i,i] + G[i+1,i+1]) sin θ
    where G_l = Right_l @ A^T @ Left_l @ P_l and i = 2k.
    """
    n_blocks = N // 2
    angles = angles_flat.reshape(L, n_blocks)

    S = build_staged_from_angles(angles, P_mats, N, L)
    cost = np.trace(A.T @ S)

    grad = np.zeros((L, n_blocks))
    for l in range(L):
        Right, Left = _compute_partial_products(angles, P_mats, N, L, l)
        G = Right @ A.T @ Left @ P_mats[l]
        for k in range(n_blocks):
            i = 2 * k
            a = G[i, i] + G[i + 1, i + 1]
            b = G[i, i + 1] - G[i + 1, i]
            theta = angles[l, k]
            grad[l, k] = b * np.cos(theta) - a * np.sin(theta)

    return -cost, -grad.ravel()


def _lbfgs_refine(A, angles, P_mats, N, L, maxiter=500):
    """Refine angles with L-BFGS (handles cross-layer coupling)."""
    result = minimize(
        _neg_cost_and_grad,
        angles.ravel(),
        args=(A, P_mats, N, L),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12},
    )
    n_blocks = N // 2
    return result.x.reshape(L, n_blocks), -result.fun


#============================================================================
#main projection entry point
#============================================================================

def project_dense_to_staged(A, N, L=3, n_restarts=20, max_iter=50, tol=1e-14,
                            seed=42, verbose=False, perms='heterogeneous'):
    """Project dense orthogonal A onto the staged b=2, L-layer manifold.

    A ≈ S(θ*) = P_{L-1} B_{L-1} ··· P_0 B_0

    Each B_l is block-diagonal 2×2 rotations; each P_l is a permutation.

    Parameters
    ----------
    A : (N, N) orthogonal matrix
    N : matrix dimension (must be even)
    L : number of staged layers (default 3)
    n_restarts : number of optimisation starts (first = greedy, rest = random)
    max_iter : max alternating sweeps per start (Phase 1)
    tol : convergence tolerance on tr(A^T S)
    seed : random seed for reproducibility
    verbose : print progress
    perms : 'heterogeneous' (shuffle+bitrev+stride per layer, default),
            'perfect_shuffle' (same PS every layer),
            or list of L permutation index arrays

    Returns
    -------
    result : dict with keys:
        'angles'      : (L, N//2) optimal rotation angles
        'perms'       : list of L permutation index arrays
        'S'           : N x N projected orthogonal matrix
        'cost'        : final tr(A^T S)  (max = N when S == A)
        'frob_error'  : ||A - S||_F
        'rel_trace'   : tr(A^T S) / N  (1.0 = perfect match)
    """
    assert N % 2 == 0, "N must be even for b=2 staged structure"
    assert np.allclose(A.T @ A, np.eye(N), atol=1e-6), "A must be orthogonal"

    n_blocks = N // 2
    rng = np.random.default_rng(seed)

    #build the permutation schedule (one permutation per layer)
    if isinstance(perms, str):
        if perms == 'heterogeneous':
            perm_list = heterogeneous_perms(N, L)
        elif perms == 'perfect_shuffle':
            perm_list = [perfect_shuffle_perm(N)] * L
        else:
            raise ValueError(f"Unknown perm type: {perms}")
    else:
        perm_list = list(perms)
    P_mats = [perm_matrix(p, N) for p in perm_list]

    best_cost = -np.inf
    best_angles = None

    #phase 1, multi-start: run the alternating closed-form from several seeds
    for restart in range(n_restarts):
        if restart == 0:
            #first start uses a greedy outside-in initialisation
            angles = _greedy_init(A, P_mats, N, L)
        else:
            #subsequent starts use random angles in [0, 2 pi)
            angles = rng.uniform(0, 2 * np.pi, (L, n_blocks))

        angles, cost = _alternating_optimize(A, angles.copy(), P_mats, N, L,
                                             max_iter, tol)
        if cost > best_cost:
            best_cost = cost
            best_angles = angles.copy()

    if verbose:
        print(f"  Phase 1 best: tr/N = {best_cost / N:.6f}")

    #l-bfgs refinement of the best alternating solution, handles cross-layer coupling
    best_angles, best_cost = _lbfgs_refine(A, best_angles, P_mats, N, L)

    if verbose:
        print(f"  Phase 2 (L-BFGS): tr/N = {best_cost / N:.6f}")

    #small perturbation restarts to escape any remaining local minima
    n_perturb = max(5, n_restarts // 4)
    for _ in range(n_perturb):
        perturbed = best_angles + rng.normal(0, 0.3, best_angles.shape)
        ref_angles, ref_cost = _lbfgs_refine(A, perturbed, P_mats, N, L,
                                             maxiter=200)
        if ref_cost > best_cost:
            best_cost = ref_cost
            best_angles = ref_angles.copy()

    S = build_staged_from_angles(best_angles, P_mats, N, L)
    frob_error = np.linalg.norm(A - S, 'fro')

    if verbose:
        print(f"  Final: tr/N = {best_cost / N:.6f}, ||A-S||_F = {frob_error:.4f}")

    return {
        'angles': best_angles,
        'perms': perm_list,
        'S': S,
        'cost': best_cost,
        'frob_error': frob_error,
        'rel_trace': best_cost / N,
    }


#============================================================================
#self-tests
#============================================================================

def test_build_roundtrip():
    """Verify build_staged_from_angles produces orthogonal matrices."""
    rng = np.random.default_rng(42)
    for N in [8, 16, 32]:
        L = 3
        angles = rng.uniform(0, 2 * np.pi, (L, N // 2))
        perm = perfect_shuffle_perm(N)
        P_mats = [perm_matrix(perm, N) for _ in range(L)]
        S = build_staged_from_angles(angles, P_mats, N, L)
        err = np.linalg.norm(S.T @ S - np.eye(N))
        assert err < 1e-12, f"N={N}: not orthogonal, err={err}"
    print("PASS: build_staged_from_angles produces orthogonal matrices")


def test_apply_matches_dense():
    """Verify O(N) apply matches dense multiply."""
    rng = np.random.default_rng(42)
    for N in [8, 16, 32]:
        L = 3
        angles = rng.uniform(0, 2 * np.pi, (L, N // 2))
        perm = perfect_shuffle_perm(N)
        P_mats = [perm_matrix(perm, N) for _ in range(L)]
        S = build_staged_from_angles(angles, P_mats, N, L)
        x = rng.standard_normal(N)
        y_dense = S @ x
        y_fast = apply_staged(x, angles, [perm] * L, N, L)
        err = np.linalg.norm(y_dense - y_fast)
        assert err < 1e-12, f"N={N}: apply mismatch, err={err}"
    print("PASS: O(N) apply matches dense multiply")


def test_gradient_vs_finite_diff():
    """Verify analytical gradient matches finite differences."""
    rng = np.random.default_rng(42)
    N, L = 8, 3
    angles = rng.uniform(0, 2 * np.pi, (L, N // 2))
    perm = perfect_shuffle_perm(N)
    P_mats = [perm_matrix(perm, N) for _ in range(L)]
    H = rng.standard_normal((N, N))
    Q, R = np.linalg.qr(H)
    A = Q @ np.diag(np.sign(np.diag(R)))

    _, grad_anal = _neg_cost_and_grad(angles.ravel(), A, P_mats, N, L)

    eps = 1e-7
    grad_fd = np.zeros_like(grad_anal)
    x0 = angles.ravel()
    for i in range(len(x0)):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        fp, _ = _neg_cost_and_grad(xp, A, P_mats, N, L)
        fm, _ = _neg_cost_and_grad(xm, A, P_mats, N, L)
        grad_fd[i] = (fp - fm) / (2 * eps)

    err = np.linalg.norm(grad_anal - grad_fd) / (np.linalg.norm(grad_anal) + 1e-15)
    assert err < 1e-6, f"Gradient mismatch: relative error = {err:.2e}"
    print(f"PASS: gradient matches finite diff (rel err = {err:.2e})")


def test_projection_identity():
    """If A is in the manifold, projection should recover it (near-zero error)."""
    rng = np.random.default_rng(42)
    for N in [8, 16]:
        L = 3
        true_angles = rng.uniform(0, 2 * np.pi, (L, N // 2))
        perm = perfect_shuffle_perm(N)
        P_mats = [perm_matrix(perm, N) for _ in range(L)]
        A = build_staged_from_angles(true_angles, P_mats, N, L)

        result = project_dense_to_staged(A, N, L=L, n_restarts=15,
                                         max_iter=50, seed=123,
                                         perms='perfect_shuffle')
        assert result['frob_error'] < 1e-4, \
            f"N={N}: in-manifold error = {result['frob_error']:.2e}"
        assert result['rel_trace'] > 1 - 1e-8, \
            f"N={N}: rel_trace = {result['rel_trace']:.10f}"
    print("PASS: projection recovers in-manifold matrices")


def test_projection_improves_on_random():
    """Projection should beat random staged initialisation."""
    rng = np.random.default_rng(42)
    N, L = 16, 3
    H = rng.standard_normal((N, N))
    Q, R = np.linalg.qr(H)
    A = Q @ np.diag(np.sign(np.diag(R)))

    result = project_dense_to_staged(A, N, L=L, n_restarts=15, seed=99)

    rand_angles = rng.uniform(0, 2 * np.pi, (L, N // 2))
    perm = perfect_shuffle_perm(N)
    P_mats = [perm_matrix(perm, N) for _ in range(L)]
    S_rand = build_staged_from_angles(rand_angles, P_mats, N, L)
    rand_cost = np.trace(A.T @ S_rand) / N

    assert result['rel_trace'] > rand_cost, \
        f"Projection ({result['rel_trace']:.4f}) should beat random ({rand_cost:.4f})"
    print(f"PASS: projection rel_trace={result['rel_trace']:.4f} > "
          f"random={rand_cost:.4f}")


def test_projection_orthogonality():
    """Projected matrix must be orthogonal."""
    rng = np.random.default_rng(42)
    for N in [8, 16, 32]:
        H = rng.standard_normal((N, N))
        Q, R = np.linalg.qr(H)
        A = Q @ np.diag(np.sign(np.diag(R)))

        result = project_dense_to_staged(A, N, L=3, n_restarts=5, seed=77)
        S = result['S']
        err = np.linalg.norm(S.T @ S - np.eye(N))
        assert err < 1e-12, f"N={N}: not orthogonal, err={err}"
    print("PASS: projected matrices are orthogonal for N=8,16,32")


def test_lsd_self_zero():
    """LSD of a matrix against itself should be zero."""
    rng = np.random.default_rng(42)
    N = 8
    delays = [7, 11, 13, 17, 19, 23, 29, 31]
    H = rng.standard_normal((N, N))
    Q, R = np.linalg.qr(H)
    A = Q @ np.diag(np.sign(np.diag(R)))

    lsd = log_spectral_distance(A, A, delays, gamma=0.999, n_freqs=512)
    assert lsd < 1e-10, f"LSD(A, A) should be 0, got {lsd:.2e}"
    print(f"PASS: LSD(A, A) = {lsd:.2e} ≈ 0")


def test_lsd_refine_improves():
    """LSD refinement should reduce LSD compared to Frobenius-only projection."""
    rng = np.random.default_rng(42)
    N = 8
    L = 3
    delays = [7, 11, 13, 17, 19, 23, 29, 31]
    gamma = 0.999

    H = rng.standard_normal((N, N))
    Q, R = np.linalg.qr(H)
    A = Q @ np.diag(np.sign(np.diag(R)))

    #frobenius projection
    result = project_dense_to_staged(A, N, L=L, n_restarts=10, seed=42)
    lsd_before = log_spectral_distance(A, result['S'], delays, gamma,
                                       n_freqs=512)

    #lsd refinement
    refined = refine_with_lsd(result, A, delays, gamma=gamma, n_freqs=512,
                              maxiter=100)

    assert refined['lsd'] <= lsd_before + 0.01, \
        f"LSD should not increase: {refined['lsd']:.2f} > {lsd_before:.2f}"
    print(f"PASS: LSD refinement {lsd_before:.2f} → {refined['lsd']:.2f} dB "
          f"(Δ = {refined['lsd'] - lsd_before:+.2f})")


if __name__ == "__main__":
    test_build_roundtrip()
    test_apply_matches_dense()
    test_gradient_vs_finite_diff()
    test_projection_identity()
    test_projection_improves_on_random()
    test_projection_orthogonality()
    test_lsd_self_zero()
    test_lsd_refine_improves()
    print("\nAll staged projection tests passed.")
