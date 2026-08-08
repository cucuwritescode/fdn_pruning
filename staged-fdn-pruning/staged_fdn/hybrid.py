"""
Hessenberg-Householder Hybrid Matrix
=====================================
A = Householder(v) @ Hessenberg(thetas)
Spectral benefits of OPUC interlacing + Temporal benefits of global mixing.

Author: Facundo Franchino
"""

import numpy as np
from experiments.unitary_hessenberg import build_unitary_hessenberg, project_dense_to_schur, unitary_hessenberg_matvec

def apply_hybrid(x, thetas, v):
    """
    Apply A = H(v) @ Hessenberg(thetas) in O(N).

    Applies the N-1 Givens rotations, then a single Householder reflection.
    """
    #apply the hessenberg chain
    y = unitary_hessenberg_matvec(thetas, x)

    #apply the householder reflection
    dot = np.dot(v, y)
    y = y - 2.0 * dot * v
    
    return y

def build_dense_hybrid(thetas, v):
    """
    Form the dense N x N matrix A = H(v) @ Hessenberg(thetas).
    """
    N = len(v)
    A_hess = build_unitary_hessenberg(thetas, N)
    H = np.eye(N) - 2.0 * np.outer(v, v)
    return H @ A_hess

def project_dense_to_hybrid(A_dense):
    """
    Project a dense orthogonal A onto the Hybrid manifold.
    Applies a small-angle (epsilon) Householder reflection
    to 'short-circuit' the Hessenberg chain.
    
    The vector v = (1-epsilon)*e0 + epsilon*eN connects the ends.
    """
    N = A_dense.shape[0]
    thetas = project_dense_to_schur(A_dense, N)
    
    #triangle vector: couples first, middle, and last nodes
    v = np.zeros(N)
    v[0] = 1.0
    v[N//2] = 0.5
    v[N-1] = 0.5
    v /= np.linalg.norm(v)
    
    return thetas, v

def test_hybrid():
    """Verify Hybrid matrix properties."""
    N = 16
    rng = np.random.default_rng(42)
    thetas = rng.uniform(0, 2*np.pi, N-1)
    v = rng.standard_normal(N)
    v /= np.linalg.norm(v)
    
    A = build_dense_hybrid(thetas, v)
    
    #orthogonality
    err = np.linalg.norm(A.T @ A - np.eye(N))
    assert err < 1e-12, f"Hybrid not orthogonal: {err}"
    
    #matvec
    x = rng.standard_normal(N)
    y_dense = A @ x
    y_fast = apply_hybrid(x, thetas, v)
    assert np.allclose(y_dense, y_fast), "Hybrid matvec mismatch"
    
    print(f"PASS: Hybrid (N={N}) verified.")

if __name__ == "__main__":
    test_hybrid()
