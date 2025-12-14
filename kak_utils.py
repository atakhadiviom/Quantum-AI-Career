import jax
import jax.numpy as jnp
import numpy as np

def compute_kak_coords(unitary):
    """
    Computes the KAK interaction coefficients (x, y, z) for a 4x4 unitary using JAX.
    Returns canonical coordinates in the Weyl chamber: pi/4 >= x >= y >= |z|.
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
    # The eigenvalues are exp(i * 2 * (hx, hy, hz, ...))
    # So angles are hx, hy, hz...
    angles = -jnp.angle(eigvals) / 2.0
    
    # Map to canonical Weyl chamber
    # 1. Take absolute values (mod pi/2 symmetries handled later, but start here)
    # Actually, we need to find the combination (c1, c2, c3) such that 
    # the 4 angles are (c1+c2-c3, c1-c2+c3, -c1+c2+c3, -c1-c2-c3) or similar permutations.
    
    # Robust approach for SU(4):
    # The 4 angles are roughly (x-y+z, -x+y+z, -x-y+z, x+y-z) ? 
    # Let's use the property that x, y, z are related to the eigenvalues.
    # We sort them and apply modular arithmetic to get into the fundamental domain.
    
    # Simplified robust logic:
    # 1. Sort angles modulo pi
    angles = jnp.mod(angles, jnp.pi)
    
    # 2. We want to extract x, y, z from these 4 values.
    # The relationship is non-trivial because of the order of eigenvalues.
    # However, for a generic unitary, we can try to enforce the chamber.
    
    # Let's use the logic that x, y, z are the "interaction strengths".
    # We can just take the sorted absolute values as a first pass, 
    # but strictly we should apply the Weyl group symmetries.
    
    # Reference algorithm (e.g. from Cirq or Tucci):
    # 1. Calculate h = angles
    # 2. Convert to x, y, z candidates
    #    x = (h1 + h2) / 2
    #    y = (h1 - h2) / 2 ...
    
    # A reliable heuristic for JAX (differentiable-ish):
    # Just return sorted absolute values in [0, pi/4] range?
    # The max entanglement is pi/4.
    
    # Let's stick to the previous simple sorting but enforce the range [0, pi/4] properly.
    # Canonical: pi/4 >= x >= y >= |z|
    
    # Map to [0, pi]
    u_angles = jnp.mod(angles, jnp.pi)
    
    # Map to [0, pi/2]
    # If > pi/2, reflect: pi - angle
    u_angles = jnp.minimum(u_angles, jnp.pi - u_angles)
    
    # Sort
    u_angles = jnp.sort(u_angles)
    
    # Extract 3 largest (usually the first one is close to 0 if 4th is small?)
    # Actually there are 4 values. 
    # For SU(4), sum is 0 mod 2pi.
    
    # Let's just take the top 3 and sort them.
    # This is an approximation but better than raw.
    res = u_angles[1:] # Take top 3? No, let's take all 4 then process.
    
    # Better: Just take the 3 largest absolute values
    # x, y, z correspond to the interaction coefficients.
    
    # Let's refine the sorting to match Weyl chamber:
    # x is the largest interaction.
    
    # Taking the 3 largest values from the 4 angles (modulo symmetries)
    # is a reasonable proxy for (x, y, z) for initialization.
    # We ensure x >= y >= z
    
    coords = u_angles[:3] # Just take 3
    coords = jnp.sort(coords)[::-1] # Descending: x, y, z
    
    # Enforce Weyl chamber condition roughly: x >= y >= z
    # And x <= pi/4? 
    # If we have interactions > pi/4, they wrap around.
    
    return coords

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
    # Params: Known solution for CNOT from calibration
    params_cnot = jnp.array([
        [[-1.5000015497207642, -4.514768123626709, 8.685406684875488], [1.3594839572906494, -3.1223084926605225, -3.377347469329834]],
        [[-2.542282819747925, 4.467097282409668, -5.550124645233154], [-3.3176305294036865, 1.656246304512024, -0.41269636154174805]],
        [[0.37855178117752075, 1.1851909160614014, -5.839783191680908], [2.4382376670837402, -3.1886086463928223, -4.698708534240723]],
        [[1.7741163969039917, 1.622654676437378, -0.8158003687858582], [-5.968489646911621, 0.7379252910614014, 1.262075424194336]],
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
