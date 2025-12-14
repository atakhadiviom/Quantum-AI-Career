import jax
import jax.numpy as jnp
import numpy as np

def compute_kak_coords(unitary):
    """
    Computes the KAK interaction coefficients (x, y, z) for a 4x4 unitary using JAX.
    """
    # Magic Basis Matrix
    inv_sqrt2 = 1.0 / jnp.sqrt(2.0)
    Q = jnp.array([
        [1, 0, 0, 1j],
        [0, 1j, 1, 0],
        [0, 1j, -1, 0],
        [1, 0, 0, -1j]
    ], dtype=jnp.complex64) * inv_sqrt2
    
    # Transform to magic basis
    m = jnp.conj(Q).T @ unitary @ Q
    
    # Compute gamma = m.T @ m
    gamma = m.T @ m
    
    # Eigenvalues of gamma
    eigvals = jnp.linalg.eigvals(gamma)
    
    # Extract angles: -angle(eigvals) / 2
    angles = -jnp.angle(eigvals) / 2.0
    
    # Map to canonical Weyl chamber (approximate)
    # This is a simplification. A full reduction requires sorting and Weyl group symmetries.
    # For initialization purposes, sorting the absolute values might be enough to get consistency.
    # We take the first 3 components which correspond to x, y, z interactions (roughly).
    # Note: eigenvalues come in conjugate pairs for SU(4) mapped to SO(4).
    # We need to be careful with extraction.
    
    # A robust extraction involves sorting.
    # Let's sort to match Cirq's convention roughly: x >= y >= z
    
    # For now, let's just return sorted absolute values to be safe against ordering permutations
    # This loses sign information but for interaction strength initialization it might be okay.
    # Better: Use the standard extraction if possible.
    
    # Simply sorting the values:
    sorted_angles = jnp.sort(jnp.abs(angles))
    # The eigenvalues of gamma are exp(2i(hx, hy, hz)). 
    # We have 4 eigenvalues. In the magic basis for SU(4), they relate to (x,y,z).
    
    # Let's trust the optimization loop to handle minor discrepancies 
    # and just provide a consistent "signature" of the unitary.
    return sorted_angles[:3]

def get_sycamore_initial_params(kak_coords):
    """
    Generates initial parameters for the 2-Sycamore ansatz based on KAK coordinates.
    
    Args:
        kak_coords: (3,) array of interaction coefficients (x, y, z)
        
    Returns:
        params: (4, 2, 3) array of parameters for the circuit
    """
    # 1. Define Anchor Points (Params for known gates)
    
    # Anchor A: Identity (Coords ~ [0, 0, 0])
    # Params: All zeros (simplest identity)
    params_id = jnp.zeros((4, 2, 3))
    
    # Anchor B: CNOT (Coords ~ [0.5, 0, 0])
    # Params: Known solution for CNOT from compilation_sycamore.py
    params_cnot = jnp.array([
        [[-0.5863459345743176, 0.5833333333333335, 0.4999999999999998], [0.5, 0.41666666666666696, 0.5]],
        [[0.0, 0.0, 0.0], [0.4771266984986657, 0.0, 0.0]],
        [[0.5863459345743176, 0.08333333333333304, -0.08333333333333348], [0.5, -0.7499999999999996, 0.24999999999999944]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ])
    
    # Anchor C: iSWAP (Coords ~ [0.5, 0.5, 0])
    # Sycamore is close to iSWAP.
    # 2 Sycamores can make iSWAP easily?
    # Let's assume params_cnot is a good start for high-entangling gates.
    
    # 2. Heuristic Selection
    # Calculate distance to anchors
    # We focus on the "strength" of entanglement which is roughly norm(coords)
    
    interaction_strength = jnp.linalg.norm(kak_coords)
    
    # If strength is low, start close to Identity.
    # If strength is high, start close to CNOT.
    
    # Simple linear interpolation based on strength
    # Max strength for CNOT is 0.5 (x=0.5). 
    # Normalized weight:
    w = jnp.clip(interaction_strength / 0.5, 0.0, 1.0)
    
    # Interpolate parameters
    # This is a naive "homotopy" initialization
    init_params = (1.0 - w) * params_id + w * params_cnot
    
    return init_params
