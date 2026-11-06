"""
radix-4 staged projection, dense O(N^2) down to b=4 constant geometry O(N*L)

projects a dense orthogonal matrix onto the staged family with 4x4
orthogonal blocks and the pease constant-geometry permutation,

    S = B_{L-1} P ... B_1 P B_0 P

each B_l is block-diagonal with N/4 orthogonal 4x4 blocks. P is the fixed
gather that reads lanes b, b+N/4, b+N/2, b+3N/4. mixing spread multiplies
by four per layer, so full spread needs 4^L >= N, half the depth of the
b=2 structure.

this is an alternative projection target to projection.py, not a
replacement. the b=2 pipeline stays as validated in the thesis, callers
pick the manifold per size on measured lsd and runtime.

two phases, mirroring projection.py. phase one is alternating closed-form
block coordinate ascent. ||A - S||_F^2 = 2N - 2 tr(A^T S), so we maximise
the trace. with the other layers held fixed,

    G_l = P Right_l A^T Left_l

and each 4x4 diagonal sub-block of G_l is a little orthogonal procrustes
problem with closed-form solution B_k = (U V^T)^T from G_k = U S V^T.
same theorem as the b=2 atan2 update, one dimension up. phase two refines
all blocks jointly with l-bfgs on a cayley chart around the phase-1
solution, B_k = C(K_k) B_k0, so every iterate stays exactly orthogonal
and keeps its determinant.

facundo franchino, 2026
"""

import numpy as np

try:
    from .projection import _freq_response_fast
except ImportError:
    from projection import _freq_response_fast


#============================================================================
#constant-geometry permutation and structure builders
#============================================================================

def pease_perm(N):
    """gather permutation for the radix-4 constant-geometry layer.

    position 4b+j reads input b + (N/4)j, so each 4x4 block mixes four
    lanes a quarter of the vector apart and the write lands contiguous.
    same geometry every layer, which is what the c++ kernel exploits.
    the shuffle is free in the load addressing, exactly as the perfect
    shuffle is free in vst2q for the b=2 kernel.
    """
    Q = N // 4
    perm = np.empty(N, dtype=int)
    for b in range(Q):
        for j in range(4):
            perm[4 * b + j] = b + Q * j
    return perm


def apply_layer(x, W):
    """one layer in O(N), out[4b+r] = sum_j W[b,r,j] x[b + Qj]."""
    Q = W.shape[0]
    #xg[j, b] = x[Q j + b], the gathered operands for every block at once
    xg = x.reshape(4, Q)
    return np.einsum('brj,jb->br', W, xg).reshape(4 * Q)


def apply_radix4(x, blocks):
    """O(N·L) matrix-vector multiply without forming the dense matrix."""
    y = x.astype(float, copy=True)
    for W in blocks:
        y = apply_layer(y, W)
    return y


def build_radix4_from_blocks(blocks, N):
    """Build S = B_{L-1} P ··· B_0 P from an (L, N/4, 4, 4) block array."""
    Q = N // 4
    S = np.eye(N)
    for W in blocks:
        #apply the layer to every column of S at once
        S = np.einsum('brj,jbk->brk', W, S.reshape(4, Q, N)).reshape(N, N)
    return S


def random_blocks(N, L, rng):
    """Random orthogonal 4×4 blocks via QR, shape (L, N/4, 4, 4)."""
    Q = N // 4
    blocks = np.empty((L, Q, 4, 4))
    for l in range(L):
        for k in range(Q):
            H = rng.standard_normal((4, 4))
            q, r = np.linalg.qr(H)
            blocks[l, k] = q @ np.diag(np.sign(np.diag(r)))
    return blocks


#============================================================================
#per-block procrustes (the closed form of the alternating phase)
#============================================================================

def _procrustes_blocks(G, Q):
    """Per-block orthogonal factors maximising tr(G B), B block-diagonal.

    For each diagonal 4×4 sub-block G_k = U Σ V^T the optimum over the
    full orthogonal group is B_k = (U V^T)^T. Reflections are permitted:
    the runtime kernel only needs orthogonality, not det = +1.
    """
    W = np.empty((Q, 4, 4))
    for k in range(Q):
        Gk = G[4 * k:4 * k + 4, 4 * k:4 * k + 4]
        U, _, Vt = np.linalg.svd(Gk)
        W[k] = (U @ Vt).T
    return W


#============================================================================
#phase 1, greedy init + alternating closed-form
#============================================================================

def _greedy_init(A, N, L, perm):
    """Greedy outside-in layer peeling, the b=4 analogue of the b=2 peel."""
    Q = N // 4
    blocks = np.zeros((L, Q, 4, 4))
    residual = A.copy()

    for l in range(L - 1, -1, -1):
        #best B_l against the residual. maximise tr(residual^T B P),
        #cyclically tr(P residual^T B), so procrustes on (P residual^T)
        X = residual.T[perm]
        blocks[l] = _procrustes_blocks(X, Q)
        #peel the layer off, residual <- Phi_l^T residual = P^T B^T residual
        M = residual.reshape(Q, 4, N)
        Bt_res = np.einsum('bjr,bjk->brk', blocks[l], M).reshape(N, N)
        residual = np.empty_like(Bt_res)
        residual[perm] = Bt_res

    return blocks


def _alternating_optimize(A, blocks, N, L, max_iter, tol, perm):
    """Alternating block coordinate ascent with exact per-block updates.

    Each layer update solves N/4 independent 4×4 procrustes problems in
    closed form; forward and backward sweeps alternate until the trace
    objective stops improving, exactly as in the b=2 projection.
    """
    Q = N // 4
    S = build_radix4_from_blocks(blocks, N)
    old_cost = np.trace(A.T @ S)

    for iteration in range(max_iter):
        for sweep in (range(L), range(L - 1, -1, -1)):
            for l in sweep:
                Right = build_radix4_from_blocks(blocks[:l], N)
                Left = build_radix4_from_blocks(blocks[l + 1:], N)
                #G_l = P Right A^T Left, the permutation as a row gather
                G = (Right @ A.T @ Left)[perm]
                blocks[l] = _procrustes_blocks(G, Q)

        S = build_radix4_from_blocks(blocks, N)
        new_cost = np.trace(A.T @ S)
        if new_cost - old_cost < tol:
            return blocks, new_cost
        old_cost = new_cost

    return blocks, old_cost


#============================================================================
#phase 2, joint cayley refinement of the trace objective
#============================================================================

_SKEW_I = np.array([0, 0, 0, 1, 1, 2])
_SKEW_J = np.array([1, 2, 3, 2, 3, 3])


def _cayley_np(theta):
    """(..., 6) skew parameters to (..., 4, 4) rotations via Cayley."""
    K = np.zeros(theta.shape[:-1] + (4, 4))
    K[..., _SKEW_I, _SKEW_J] = theta
    K = K - np.swapaxes(K, -1, -2)
    I4 = np.eye(4)
    return np.linalg.solve(I4 + K, I4 - K)


def _cayley_refine_trace(A, blocks, N, L, perm, maxiter=200):
    """Refine all blocks jointly with L-BFGS on a Cayley chart.

    Parametrises B = C(K) B0 around the phase-1 blocks, so every iterate
    is exactly orthogonal and block determinants are preserved. L-BFGS
    with a strong-Wolfe line search adjusts interactions between layers,
    as the L-BFGS phase does for the b=2 projection.
    """
    import torch

    Q = N // 4
    A_t = torch.tensor(A, dtype=torch.float64)
    B0 = torch.tensor(blocks, dtype=torch.float64)
    theta = torch.zeros(L, Q, 6, dtype=torch.float64, requires_grad=True)
    idx_i = torch.tensor(_SKEW_I)
    idx_j = torch.tensor(_SKEW_J)
    I4 = torch.eye(4, dtype=torch.float64)

    def build(th):
        K = torch.zeros(L, Q, 4, 4, dtype=torch.float64)
        K[..., idx_i, idx_j] = th
        K = K - K.transpose(-1, -2)
        B = torch.linalg.solve(I4 + K, I4 - K) @ B0
        S = torch.eye(N, dtype=torch.float64)
        for l in range(L):
            S = torch.einsum('brj,jbk->brk', B[l], S.reshape(4, Q, N)).reshape(N, N)
        return S

    optimizer = torch.optim.LBFGS(
        [theta], max_iter=maxiter, history_size=30,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-12, tolerance_change=1e-14)

    def closure():
        optimizer.zero_grad()
        loss = -torch.trace(A_t.T @ build(theta))
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        cost = float(torch.trace(A_t.T @ build(theta)))
    refined = _cayley_np(theta.detach().numpy()) @ blocks
    return refined, cost


#============================================================================
#main projection entry point
#============================================================================

def project_dense_to_radix4(A, N, L=3, n_restarts=10, max_iter=50, tol=1e-14,
                            seed=42, verbose=False, refine_iters=200):
    """project dense orthogonal A onto the radix-4, L-layer family.

    A ~ S = B_{L-1} P ... B_0 P

    each B_l is block-diagonal orthogonal 4x4, P is the fixed pease
    constant-geometry permutation. full structural spread needs 4^L >= N.

    Parameters
    ----------
    A : (N, N) orthogonal matrix
    N : matrix dimension (must be divisible by 4)
    L : number of layers (default 3)
    n_restarts : optimisation starts (first = greedy, rest = random)
    max_iter : max alternating sweeps per start
    tol : convergence tolerance on tr(A^T S)
    seed : random seed for reproducibility
    verbose : print progress
    refine_iters : L-BFGS iterations for joint Cayley refinement (0 = skip)

    Returns
    -------
    result : dict with keys:
        'blocks'     : (L, N/4, 4, 4) orthogonal blocks
        'perm'       : the Pease permutation index array
        'S'          : N × N projected orthogonal matrix
        'cost'       : final tr(A^T S)  (max = N when S == A)
        'frob_error' : ||A - S||_F
        'rel_trace'  : tr(A^T S) / N  (1.0 = perfect match)
    """
    assert N % 4 == 0, "N must be divisible by 4 for the radix-4 structure"
    assert np.allclose(A.T @ A, np.eye(N), atol=1e-6), "A must be orthogonal"

    perm = pease_perm(N)
    rng = np.random.default_rng(seed)

    best_cost = -np.inf
    best_blocks = None

    for restart in range(n_restarts):
        if restart == 0:
            blocks = _greedy_init(A, N, L, perm)
        else:
            blocks = random_blocks(N, L, rng)

        blocks, cost = _alternating_optimize(A, blocks, N, L, max_iter, tol, perm)
        if cost > best_cost:
            best_cost = cost
            best_blocks = blocks.copy()

    if verbose:
        print(f"  Phase 1 best: tr/N = {best_cost / N:.6f}")

    if refine_iters:
        best_blocks, best_cost = _cayley_refine_trace(
            A, best_blocks, N, L, perm, maxiter=refine_iters)
        if verbose:
            print(f"  Phase 2 (Cayley/L-BFGS): tr/N = {best_cost / N:.6f}")

    S = build_radix4_from_blocks(best_blocks, N)
    frob_error = np.linalg.norm(A - S, 'fro')

    if verbose:
        print(f"  Final: tr/N = {best_cost / N:.6f}, ||A-S||_F = {frob_error:.4f}")

    return {
        'blocks': best_blocks,
        'perm': perm,
        'S': S,
        'cost': best_cost,
        'frob_error': frob_error,
        'rel_trace': best_cost / N,
    }


#============================================================================
#phase 3, log-spectral refinement (port of projection.refine_with_lsd)
#============================================================================

def refine_with_lsd_radix4(result, A_dense, delays, gamma=1.0, b=None, c=None,
                           n_freqs=512, maxiter=200, lr=1e-2, verbose=False,
                           precision='double'):
    """refine radix-4 blocks against the fdn magnitude response (lsd).

    loss and the batched frequency-response solve are the same as
    refine_with_lsd, only the S-builder differs. blocks stay orthogonal
    through the same cayley chart as the trace refinement.

    Parameters
    ----------
    result : dict from project_dense_to_radix4
    A_dense : (N, N) original dense orthogonal matrix
    delays : length-N delay line lengths
    gamma : per-sample decay (default 1.0 = lossless; use < 1 for finite H)
    b, c : input/output vectors
    n_freqs : frequency resolution for optimisation
    maxiter : optimisation steps
    lr : Adam learning rate
    verbose : print progress
    precision : 'double' (default) or 'single'; single runs the batched
        solve in complex64, roughly 3x faster at large N, and the final
        metrics are recomputed at double precision either way

    Returns
    -------
    refined : dict with same keys as project_dense_to_radix4 plus 'lsd',
              'lsd_init'
    """
    import torch

    N = A_dense.shape[0]
    blocks0 = result['blocks']
    L = blocks0.shape[0]
    Q = N // 4
    delays_arr = np.asarray(delays, dtype=float)

    if b is None:
        b = np.array([(-1.0)**i for i in range(N)]) / np.sqrt(N)
    if c is None:
        c = np.array([(-1.0)**(i // 2) for i in range(N)]) / np.sqrt(N)

    gamma_m = gamma ** delays_arr

    #working dtypes; the batched solve dominates the step
    cdtype = torch.complex64 if precision == 'single' else torch.complex128
    rdtype = torch.float32 if precision == 'single' else torch.float64

    omega = np.linspace(0, np.pi, n_freqs, endpoint=False)
    phases_np = np.exp(-1j * np.outer(omega, delays_arr))

    #constants that do not change during optimisation
    phases_t = torch.tensor(phases_np, dtype=cdtype)
    gamma_m_t = torch.tensor(gamma_m, dtype=rdtype)
    b_t = torch.tensor(b, dtype=cdtype)
    c_t = torch.tensor(c, dtype=cdtype)
    eye_t = torch.eye(N, dtype=cdtype).unsqueeze(0)

    #dense reference frequency response, computed once with no gradient
    A_dense_t = torch.tensor(A_dense, dtype=cdtype)
    col_mult_ref = gamma_m_t.unsqueeze(0).to(cdtype) * phases_t
    M_ref = eye_t - A_dense_t.unsqueeze(0) * col_mult_ref.unsqueeze(1)
    p_ref = torch.linalg.solve(M_ref, b_t.expand(n_freqs, N))
    H_ref = torch.sum(c_t.unsqueeze(0) * phases_t * p_ref, dim=1)
    H_ref_db = (20.0 * torch.log10(torch.abs(H_ref) + 1e-30)).detach()

    #cayley deltas around the projected blocks are the trainable parameters
    B0 = torch.tensor(blocks0, dtype=rdtype)
    theta = torch.nn.Parameter(torch.zeros(L, Q, 6, dtype=rdtype))
    idx_i = torch.tensor(_SKEW_I)
    idx_j = torch.tensor(_SKEW_J)
    I4 = torch.eye(4, dtype=rdtype)
    optimizer = torch.optim.Adam([theta], lr=lr)

    def forward(th):
        #build s = b_{l-1} p ... b_0 p with orthogonal blocks c(k) b0
        K = torch.zeros(L, Q, 4, 4, dtype=rdtype)
        K[..., idx_i, idx_j] = th
        K = K - K.transpose(-1, -2)
        B = torch.linalg.solve(I4 + K, I4 - K) @ B0
        S = torch.eye(N, dtype=rdtype)
        for l in range(L):
            S = torch.einsum('brj,jbk->brk', B[l], S.reshape(4, Q, N)).reshape(N, N)
        #staged frequency response, same construction as the dense one
        Sc = S.to(cdtype)
        col_mult = gamma_m_t.unsqueeze(0).to(cdtype) * phases_t
        M = eye_t - Sc.unsqueeze(0) * col_mult.unsqueeze(1)
        p = torch.linalg.solve(M, b_t.expand(n_freqs, N))
        H = torch.sum(c_t.unsqueeze(0) * phases_t * p, dim=1)
        H_db = 20.0 * torch.log10(torch.abs(H) + 1e-30)
        lsd = torch.sqrt(torch.mean((H_ref_db - H_db) ** 2))
        return lsd

    with torch.no_grad():
        lsd_init = float(forward(theta))
    if verbose:
        print(f"  LSD before refinement: {lsd_init:.2f} dB")

    best_lsd = lsd_init
    best_theta = theta.detach().clone()
    for step in range(maxiter):
        optimizer.zero_grad()
        lsd = forward(theta)
        lsd_val = float(lsd.detach())
        #Save the coordinates that produced this loss, before updating them.
        if lsd_val < best_lsd:
            best_lsd = lsd_val
            best_theta = theta.detach().clone()
        lsd.backward()
        optimizer.step()
        if verbose and (step + 1) % 50 == 0:
            print(f"    step {step+1}/{maxiter}: LSD = {lsd_val:.4f} dB")

    with torch.no_grad():
        if float(forward(theta)) < best_lsd:
            best_theta = theta.detach().clone()
    #rebuild the best blocks and score everything at double precision
    blocks_final = _cayley_np(best_theta.to(torch.float64).numpy()) @ blocks0
    S_refined = build_radix4_from_blocks(blocks_final, N)
    frob_error = np.linalg.norm(A_dense - S_refined, 'fro')

    H_ref_np = _freq_response_fast(A_dense, delays_arr, gamma_m, b, c, phases_np)
    H_s_np = _freq_response_fast(S_refined, delays_arr, gamma_m, b, c, phases_np)
    H_ref_db_np = 20 * np.log10(np.abs(H_ref_np) + 1e-30)
    H_s_db_np = 20 * np.log10(np.abs(H_s_np) + 1e-30)
    lsd_final = float(np.sqrt(np.mean((H_ref_db_np - H_s_db_np) ** 2)))

    if verbose:
        print(f"  LSD after refinement:  {lsd_final:.2f} dB "
              f"(Δ = {lsd_final - lsd_init:+.2f})")

    cost = float(np.trace(A_dense.T @ S_refined))
    return {
        'blocks': blocks_final,
        'perm': result['perm'],
        'S': S_refined,
        'lsd': lsd_final,
        'lsd_init': lsd_init,
        'frob_error': frob_error,
        'cost': cost,
        'rel_trace': cost / N,
    }


#============================================================================
#self-tests
#============================================================================

def test_build_roundtrip():
    """Random blocks must yield an exactly orthogonal S."""
    rng = np.random.default_rng(42)
    for N in [8, 16, 32, 64]:
        L = 3
        blocks = random_blocks(N, L, rng)
        S = build_radix4_from_blocks(blocks, N)
        err = np.linalg.norm(S.T @ S - np.eye(N))
        assert err < 1e-12, f"N={N}: not orthogonal, err={err}"
    print("PASS: build_radix4_from_blocks produces orthogonal matrices")


def test_apply_matches_dense():
    """O(N·L) apply must match the dense multiply."""
    rng = np.random.default_rng(42)
    for N in [8, 16, 32]:
        L = 3
        blocks = random_blocks(N, L, rng)
        S = build_radix4_from_blocks(blocks, N)
        x = rng.standard_normal(N)
        err = np.linalg.norm(S @ x - apply_radix4(x, blocks))
        assert err < 1e-12, f"N={N}: apply mismatch, err={err}"
    print("PASS: O(N·L) apply matches dense multiply")


def test_full_mixing():
    """With 4^L >= N every output must depend on every input."""
    rng = np.random.default_rng(42)
    for N, L in [(16, 2), (32, 3), (64, 3), (256, 4)]:
        blocks = random_blocks(N, L, rng)
        S = build_radix4_from_blocks(blocks, N)
        n_zero = int(np.sum(np.abs(S) < 1e-12))
        assert n_zero == 0, f"N={N}, L={L}: {n_zero} structurally zero entries"
    print("PASS: full mixing at 4^L >= N (support of S has no zeros)")


def test_projection_identity():
    """In-manifold matrices must be recovered near-exactly."""
    rng = np.random.default_rng(42)
    for N in [16, 32]:
        L = 2
        true_blocks = random_blocks(N, L, rng)
        A = build_radix4_from_blocks(true_blocks, N)
        result = project_dense_to_radix4(A, N, L=L, n_restarts=8,
                                         max_iter=50, seed=123,
                                         refine_iters=100)
        assert result['frob_error'] < 1e-4, \
            f"N={N}: in-manifold error = {result['frob_error']:.2e}"
        assert result['rel_trace'] > 1 - 1e-8, \
            f"N={N}: rel_trace = {result['rel_trace']:.10f}"
    print("PASS: projection recovers in-manifold matrices")


def test_projection_improves_on_random():
    """Projection must beat a random radix-4 initialisation."""
    rng = np.random.default_rng(42)
    N, L = 16, 2
    H = rng.standard_normal((N, N))
    Q, R = np.linalg.qr(H)
    A = Q @ np.diag(np.sign(np.diag(R)))

    result = project_dense_to_radix4(A, N, L=L, n_restarts=8, seed=99)
    rand_cost = np.trace(A.T @ build_radix4_from_blocks(
        random_blocks(N, L, rng), N)) / N

    assert result['rel_trace'] > rand_cost, \
        f"Projection ({result['rel_trace']:.4f}) should beat random ({rand_cost:.4f})"
    print(f"PASS: projection rel_trace={result['rel_trace']:.4f} > "
          f"random={rand_cost:.4f}")


def test_projection_orthogonality():
    """The projected matrix must be orthogonal."""
    rng = np.random.default_rng(42)
    for N in [16, 32]:
        H = rng.standard_normal((N, N))
        Q, R = np.linalg.qr(H)
        A = Q @ np.diag(np.sign(np.diag(R)))
        result = project_dense_to_radix4(A, N, L=3, n_restarts=4, seed=77)
        err = np.linalg.norm(result['S'].T @ result['S'] - np.eye(N))
        assert err < 1e-12, f"N={N}: not orthogonal, err={err}"
    print("PASS: projected matrices are orthogonal for N=16,32")


def test_matches_or_beats_b2():
    """At equal spread the b=4 family is at least as expressive as b=2."""
    try:
        from .projection import project_dense_to_staged
    except ImportError:
        from projection import project_dense_to_staged
    rng = np.random.default_rng(42)
    N = 16
    H = rng.standard_normal((N, N))
    Q, R = np.linalg.qr(H)
    A = Q @ np.diag(np.sign(np.diag(R)))

    r2 = project_dense_to_staged(A, N, L=4, n_restarts=8, seed=5)
    r4 = project_dense_to_radix4(A, N, L=2, n_restarts=8, seed=5)
    assert r4['rel_trace'] > r2['rel_trace'] - 0.02, \
        f"radix-4 ({r4['rel_trace']:.4f}) far below b=2 ({r2['rel_trace']:.4f})"
    print(f"PASS: radix-4 L=2 rel_trace={r4['rel_trace']:.4f} vs "
          f"b=2 L=4 {r2['rel_trace']:.4f} at N=16")


def test_lsd_refine_improves():
    """LSD refinement must not worsen the log-spectral distance."""
    rng = np.random.default_rng(42)
    N, L = 16, 2
    delays = [149, 211, 263, 293, 337, 373, 401, 431,
              149 + 2, 211 + 2, 263 + 2, 293 + 2, 337 + 2, 373 + 2, 401 + 2, 431 + 2]
    H = rng.standard_normal((N, N))
    Q, R = np.linalg.qr(H)
    A = Q @ np.diag(np.sign(np.diag(R)))

    result = project_dense_to_radix4(A, N, L=L, n_restarts=6, seed=42)
    refined = refine_with_lsd_radix4(result, A, delays, gamma=0.999,
                                     n_freqs=256, maxiter=100)
    assert refined['lsd'] <= refined['lsd_init'] + 0.01, \
        f"LSD should not increase: {refined['lsd']:.2f} > {refined['lsd_init']:.2f}"
    err = np.linalg.norm(refined['S'].T @ refined['S'] - np.eye(N))
    assert err < 1e-12, f"refined S not orthogonal, err={err}"
    print(f"PASS: LSD refinement {refined['lsd_init']:.2f} → {refined['lsd']:.2f} dB "
          f"(Δ = {refined['lsd'] - refined['lsd_init']:+.2f})")


if __name__ == "__main__":
    test_build_roundtrip()
    test_apply_matches_dense()
    test_full_mixing()
    test_projection_identity()
    test_projection_improves_on_random()
    test_projection_orthogonality()
    test_matches_or_beats_b2()
    test_lsd_refine_improves()
    print("\nAll radix-4 projection tests passed.")
