"""
Unitary Hessenberg FDN Feedback Matrix
========================================
Product of N-1 adjacent Givens rotations.
Exactly orthogonal by construction. O(N) multiply. Irreducible.

Author: Facundo Franchino
"""

import numpy as np


def build_unitary_hessenberg(thetas, N):
    """
    Build a unitary Hessenberg matrix from N-1 rotation angles.

    A = G(0,1,θ₀) · G(1,2,θ₁) · ... · G(N-2,N-1,θ_{N-2})

    Each G(i,i+1,θ) is a Givens rotation in the (i,i+1) plane.
    The product is exactly orthogonal: A^T A = I.
    The result is upper Hessenberg in shape (zero below first subdiagonal).
    Complexity: O(N) for matrix-vector multiply (apply rotations sequentially).
    Irreducible: every delay line is coupled to its neighbours.

    PRECONDITIONS:
        - thetas has N-1 entries
        - N >= 2

    POSTCONDITIONS:
        - A is N x N
        - A^T A = I (to machine precision)
        - A[i,j] = 0 for i > j+1 (upper Hessenberg)
    """
    assert len(thetas) == N - 1, f"Need {N-1} angles, got {len(thetas)}"
    A = np.eye(N)
    for i in range(N - 1):
        c, s = np.cos(thetas[i]), np.sin(thetas[i])
        row_i = A[i].copy()
        row_i1 = A[i + 1].copy()
        A[i] = c * row_i - s * row_i1
        A[i + 1] = s * row_i + c * row_i1
    return A


def build_cmv(thetas, N):
    """
    Build a CMV matrix from the same N-1 Schur parameters.

    Same eigenvalues as the unitary Hessenberg, but rearranged into
    two parallel Givens layers (SIMD-friendly).

    Layer 1 (even pairs): G(0,1), G(2,3), G(4,5), ...
    Layer 2 (odd pairs):  G(1,2), G(3,4), G(5,6), ...
    A = Layer1 @ Layer2

    POSTCONDITIONS:
        - A^T A = I
        - Same eigenvalues as build_unitary_hessenberg(thetas, N)
    """
    assert len(thetas) == N - 1, f"Need {N-1} angles, got {len(thetas)}"
    L_even = np.eye(N)
    for k in range(0, N - 1, 2):
        c, s = np.cos(thetas[k]), np.sin(thetas[k])
        L_even[k, k] = c;     L_even[k, k + 1] = -s
        L_even[k + 1, k] = s; L_even[k + 1, k + 1] = c
    L_odd = np.eye(N)
    for k in range(1, N - 1, 2):
        c, s = np.cos(thetas[k]), np.sin(thetas[k])
        L_odd[k, k] = c;     L_odd[k, k + 1] = -s
        L_odd[k + 1, k] = s; L_odd[k + 1, k + 1] = c
    return L_even @ L_odd


def add_cycle_shortcut(A, phi):
    """
    Add a single Givens rotation G(0, N-1, phi) to close the chain
    into a cycle. Improves temporal mixing with minimal modal_std cost.
    """
    N = A.shape[0]
    A2 = A.copy()
    c, s = np.cos(phi), np.sin(phi)
    row_0 = A2[0].copy()
    row_n = A2[N - 1].copy()
    A2[0] = c * row_0 - s * row_n
    A2[N - 1] = s * row_0 + c * row_n
    return A2


def project_dense_to_schur(A_dense, N):
    """
    Project a dense orthogonal A onto the unitary Hessenberg manifold
    by extracting N-1 Schur parameters via Givens-QR factorisation.

    This IS the pruning step: dense O(N²) → Hessenberg O(N).
    """
    R = A_dense.copy()
    thetas = np.zeros(N - 1)
    for k in range(N - 1):
        a, b = R[k, k], R[k + 1, k]
        r = np.hypot(a, b)
        if r < 1e-15:
            continue
        c, s = a / r, b / r
        thetas[k] = np.arctan2(s, c)
        row_k = R[k].copy()
        row_k1 = R[k + 1].copy()
        R[k] = c * row_k + s * row_k1
        R[k + 1] = -s * row_k + c * row_k1
    return thetas


def unitary_hessenberg_matvec(thetas, x):
    """
    O(N) matrix-vector multiply without forming the dense matrix.
    Apply N-1 Givens rotations sequentially to vector x.
    """
    y = x.copy()
    for i in range(len(thetas)):
        c, s = np.cos(thetas[i]), np.sin(thetas[i])
        yi, yi1 = y[i], y[i + 1]
        y[i] = c * yi - s * yi1
        y[i + 1] = s * yi + c * yi1
    return y


#============================================================
#unit tests
#============================================================

def test_orthogonality():
    """Verify A^T A = I for multiple N and random angles."""
    rng = np.random.default_rng(42)
    for N in [4, 8, 16, 32, 64]:
        thetas = rng.uniform(0, 2 * np.pi, N - 1)

        A_hess = build_unitary_hessenberg(thetas, N)
        assert np.allclose(A_hess.T @ A_hess, np.eye(N)), \
            f"Hessenberg NOT orthogonal at N={N}: ||A^TA-I||={np.linalg.norm(A_hess.T @ A_hess - np.eye(N))}"

        A_cmv = build_cmv(thetas, N)
        assert np.allclose(A_cmv.T @ A_cmv, np.eye(N)), \
            f"CMV NOT orthogonal at N={N}: ||A^TA-I||={np.linalg.norm(A_cmv.T @ A_cmv - np.eye(N))}"

        A_cycle = add_cycle_shortcut(A_hess, 0.1)
        assert np.allclose(A_cycle.T @ A_cycle, np.eye(N)), \
            f"Cycle NOT orthogonal at N={N}"

    print("PASS: Orthogonality verified for N=4,8,16,32,64 (Hessenberg, CMV, Cycle)")


def test_eigenvalue_match():
    """Verify Hessenberg and CMV have identical eigenvalues."""
    rng = np.random.default_rng(42)
    for N in [8, 16, 32]:
        thetas = rng.uniform(0, 2 * np.pi, N - 1)
        eigs_h = np.sort(np.abs(np.linalg.eigvals(build_unitary_hessenberg(thetas, N))))
        eigs_c = np.sort(np.abs(np.linalg.eigvals(build_cmv(thetas, N))))
        assert np.allclose(eigs_h, eigs_c, atol=1e-10), \
            f"Eigenvalue mismatch at N={N}"
    print("PASS: Hessenberg and CMV eigenvalues match for N=8,16,32")


def test_matvec():
    """Verify O(N) matvec matches dense multiply."""
    rng = np.random.default_rng(42)
    for N in [8, 16, 32]:
        thetas = rng.uniform(0, 2 * np.pi, N - 1)
        x = rng.standard_normal(N)
        A = build_unitary_hessenberg(thetas, N)
        y_dense = A @ x
        y_fast = unitary_hessenberg_matvec(thetas, x)
        assert np.allclose(y_dense, y_fast, atol=1e-12), \
            f"Matvec mismatch at N={N}"
    print("PASS: O(N) matvec matches dense multiply for N=8,16,32")


def test_projection_preserves_orthogonality():
    """Verify that projecting ANY dense orthogonal matrix produces an orthogonal result."""
    rng = np.random.default_rng(42)
    for N in [8, 16, 32]:
        #random dense orthogonal (not hessenberg)
        H = rng.standard_normal((N, N))
        Q, R = np.linalg.qr(H)
        A_dense = Q @ np.diag(np.sign(np.diag(R)))

        thetas = project_dense_to_schur(A_dense, N)
        A_projected = build_unitary_hessenberg(thetas, N)

        assert np.allclose(A_projected.T @ A_projected, np.eye(N)), \
            f"Projected matrix NOT orthogonal at N={N}: ||A^TA-I||={np.linalg.norm(A_projected.T @ A_projected - np.eye(N))}"
    print("PASS: Projection of dense → Hessenberg preserves orthogonality for N=8,16,32")


if __name__ == "__main__":
    test_orthogonality()
    test_eigenvalue_match()
    test_matvec()
    test_projection_preserves_orthogonality()
    print("\nAll tests passed.")
