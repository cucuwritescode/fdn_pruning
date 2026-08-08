"""Structured pruning of Feedback Delay Networks via staged orthogonal factorisation.

A trained dense orthogonal feedback matrix is projected onto a *staged* family
S = P B_{L-1} ... P B_0 of permuted 2x2-rotation layers, which is orthogonal by
construction and applies in O(N L) instead of O(N^2) per sample.

Public API
----------
project_dense_to_staged : closed-form Frobenius projection (Phase 1)
refine_with_lsd         : log-spectral refinement of the angles (Phase 2)
build_staged_from_angles, apply_staged : construct / apply a staged matrix
fdn_frequency_response, log_spectral_distance : FDN evaluation helpers
perfect_shuffle_perm, perm_matrix, build_block_diag_rotations : building blocks
"""

from .projection import (
    project_dense_to_staged,
    refine_with_lsd,
    build_staged_from_angles,
    apply_staged,
    build_block_diag_rotations,
    fdn_frequency_response,
    log_spectral_distance,
    perfect_shuffle_perm,
    bit_reversal_perm,
    prime_stride_perm,
    heterogeneous_perms,
    perm_matrix,
)

__all__ = [
    "project_dense_to_staged",
    "refine_with_lsd",
    "build_staged_from_angles",
    "apply_staged",
    "build_block_diag_rotations",
    "fdn_frequency_response",
    "log_spectral_distance",
    "perfect_shuffle_perm",
    "bit_reversal_perm",
    "prime_stride_perm",
    "heterogeneous_perms",
    "perm_matrix",
]

__version__ = "0.1.0"
