"""export repeated-perfect-shuffle pairwise fits to the c++ BP convention.

the python projection applies P B per layer, StagedOrtho applies B P.
the two products relate by C = P.T @ S @ P, so reordering every delay-line
parameter by the same basis change keeps the complete fdn response.
"""
import numpy as np
from .projection import perfect_shuffle_perm


def export_pairwise_constant_geometry(result, delays, input_gains, output_gains):
    """kernel coefficients plus consistently relabelled fdn parameters.

    output channel q corresponds to original channel channel_order[q]. if a
    caller keeps separate per-line attenuation or filter parameters, those
    need reordering by channel_order too. delay-proportional attenuation
    already follows from the returned delays, and the scalar direct gain is
    untouched. heterogeneous schedules get rejected on purpose, the kernel
    cannot honour them.
    """
    angles = np.asarray(result['angles'], dtype=np.float64)
    if angles.ndim != 2 or angles.shape[0] < 1 or angles.shape[1] < 1:
        raise ValueError('angles must have shape (L, N/2) with positive dimensions')
    n = 2 * angles.shape[1]
    perm = perfect_shuffle_perm(n)
    schedules = result['perms']
    if len(schedules) != len(angles) or any(not np.array_equal(p, perm) for p in schedules):
        raise ValueError('the C++ kernel requires the same perfect shuffle at every layer')
    order = np.argsort(perm)
    values = [np.asarray(v).reshape(-1) for v in (delays, input_gains, output_gains)]
    if any(v.shape != (n,) for v in values):
        raise ValueError('delays and gain vectors must each contain N values')
    return dict(cos=np.cos(angles).astype(np.float32),
                sin=np.sin(angles).astype(np.float32),
                delays=values[0][order].copy(),
                input_gains=values[1][order].copy(),
                output_gains=values[2][order].copy(),
                channel_order=order)
