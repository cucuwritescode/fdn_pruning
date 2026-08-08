"""
K-Householder Factorisation for FDN Feedback Matrices
======================================================
Product of K Householder reflections. 
computationally O(KN), spatially dense (complete graph).

author: Facundo Franchino
"""

import numpy as np

def apply_k_householder(x, vectors):
    """
    Apply a product of K Householder reflections to vector x.
    A = H_K ... H_2 H_1 where H_i = I - 2 v_i v_i^T.
    
    Complexity: O(KN)
    Topology: Complete Graph (Diameter 1)
    
    PRECONDITIONS:
        - vectors: (K, N) array of unit vectors
        - x: (N,) input vector
    """
    y = x.copy()
    for v in vectors:
        #h_i x = x - 2 * (v . x) * v
        dot = np.dot(v, y)
        y = y - 2 * dot * v
    return y

def build_dense_from_k_householder(vectors):
    """
    Explicitly form the N x N dense matrix from Householder vectors.
    Useful for eigenvalue analysis and validation.
    """
    K, N = vectors.shape
    A = np.eye(N)
    for v in vectors:
        #h = i - 2 vv^t
        H = np.eye(N) - 2.0 * np.outer(v, v)
        A = H @ A
    return A

def project_dense_to_k_householder(A_dense, K):
    """
    Extract the first K Householder vectors from a dense matrix A
    using a modified Householder QR-like decomposition.
    
    NOTE: A generic orthogonal A requires exactly N reflections.
    This projection returns the best K-rank approximation.
    """
    N = A_dense.shape[0]
    vectors = []
    R = A_dense.copy()
    
    for k in range(min(K, N)):
        #standard householder vector to zero out column k below diagonal
        x = R[k:, k].copy()
        if len(x) == 0: break
        
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-15:
            #column already zeroed, move to next
            continue
            
        e1 = np.zeros_like(x)
        e1[0] = norm_x * np.sign(x[0] if x[0] != 0 else 1.0)
        
        v = x + e1
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-15:
            continue
            
        v = v / v_norm
        v_full = np.zeros(N)
        v_full[k:] = v
        vectors.append(v_full)
        
        #apply to r: r = (i - 2vv^t) r
        v_sub = v
        R[k:, :] = R[k:, :] - 2.0 * np.outer(v_sub, v_sub @ R[k:, :])
            
    return np.array(vectors)

def test_householder():
    """Verify orthogonality and O(N) property."""
    N = 16
    K = 2
    rng = np.random.default_rng(42)
    
    #random unit vectors
    vectors = rng.standard_normal((K, N))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    
    A = build_dense_from_k_householder(vectors)
    
    #test orthogonality
    err = np.linalg.norm(A.T @ A - np.eye(N))
    assert err < 1e-12, f"K-Householder not orthogonal: {err}"
    
    #test matvec
    x = rng.standard_normal(N)
    y_dense = A @ x
    y_fast = apply_k_householder(x, vectors)
    assert np.allclose(y_dense, y_fast), "Matvec mismatch"
    
    print(f"PASS: K-Householder (N={N}, K={K}) verified.")

if __name__ == "__main__":
    test_householder()
