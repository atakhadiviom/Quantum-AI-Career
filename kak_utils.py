import jax
import jax.numpy as jnp
import numpy as np

def compute_kak_coords(unitary):
    """
    Computes KAK interaction coefficients (x, y, z) using a robust JAX-compatible analytical method.
    This avoids Python callbacks to Cirq, enabling full JIT/VMAP acceleration.
    
    The logic extracts the canonical coordinates from the spectra of the Magic Basis representation,
    ensuring invariance to global phase and handling the Weyl chamber symmetries.
    """
    # Magic Basis Matrix (normalized)
    inv_sqrt2 = 1.0 / jnp.sqrt(2.0)
    Q = jnp.array([
        [1, 0, 0, 1j],
        [0, 1j, 1, 0],
        [0, 1j, -1, 0],
        [1, 0, 0, -1j]
    ], dtype=jnp.complex64) * inv_sqrt2
    
    # 1. Normalize to SU(4) to fix global phase (up to 4th root)
    # This is crucial to distinguish gates like Identity (trace 4) vs SWAP (trace 4i).
    det = jnp.linalg.det(unitary)
    u_su4 = unitary * (det ** (-0.25))
    
    # 2. Magic Basis Transform
    # m = Q^dag @ U @ Q
    m = jnp.conj(Q).T @ u_su4 @ Q
    
    # 3. Gamma Matrix (m.T @ m)
    # Eigenvalues of Gamma encode the interaction coefficients.
    gamma = m.T @ m
    eigvals = jnp.linalg.eigvals(gamma)
    
    # 4. Extract Angles (Robustly via Gap Detection)
    phases = jnp.angle(eigvals)
    phases = jnp.sort(phases)
    
    # Compute gaps between consecutive phases on the circle
    gaps = jnp.concatenate([
        phases[1:] - phases[:-1],
        jnp.array([2*jnp.pi - (phases[-1] - phases[0])])
    ])
    
    # Find the largest gap to place the branch cut
    max_gap_idx = jnp.argmax(gaps)
    
    # Calculate the center of the largest gap
    p1 = phases[max_gap_idx]
    p2_idx = (max_gap_idx + 1) % 4
    p2 = phases[p2_idx]
    is_wrap_gap = max_gap_idx == 3
    
    gap_center = jnp.where(is_wrap_gap,
                           (p1 + p2 + 2*jnp.pi) / 2.0,
                           (p1 + p2) / 2.0)
                           
    # Rotate spectrum so gap center aligns with pi
    shift = jnp.pi - gap_center
    eigvals_aligned = eigvals * jnp.exp(1j * shift)
    
    # Extract angles (continuous)
    base_angles = -jnp.angle(eigvals_aligned) / 2.0 + shift / 2.0
    
    # 5. Compute Coordinates
    v = jnp.sort(base_angles)
    
    x = (v[3] + v[2]) / 2.0
    y = (v[3] + v[1]) / 2.0
    z = (v[2] + v[1]) / 2.0
    
    coords = jnp.array([x, y, z])
    
    # 6. Fold into Weyl Chamber
    # The Weyl chamber is symmetric under shifts of pi/2 for each coordinate.
    # We map x, y, z into [0, pi/4] by folding.
    # Formula: abs( (val - pi/4) % (pi/2) - pi/4 )
    # This maps 0->0, pi/4->pi/4, pi/2->0, 3pi/4->pi/4, etc.
    
    coords = jnp.abs((coords - jnp.pi/4) % (jnp.pi/2) - jnp.pi/4)
    
    # Sort descending x >= y >= z
    coords = jnp.sort(coords)[::-1]
    
    return coords

def kak_coords_batch(unitaries):
    """Batch version of compute_kak_coords."""
    return jax.vmap(compute_kak_coords)(unitaries)

def classify_gate_type(kak_coords, atol=1e-3):
    """
    Classifies the gate type based on KAK coordinates.
    
    Args:
        kak_coords: (3,) array of interaction coefficients (x, y, z)
        atol: Absolute tolerance for comparison
        
    Returns:
        str: 'identity', 'cnot', 'swap', 'iswap', or 'generic'
    """
    # Ensure coords are positive for classification
    coords = jnp.abs(kak_coords)
    
    if jnp.all(coords < atol):
        return 'identity'
        
    # CNOT: (pi/4, 0, 0)
    if jnp.abs(coords[0] - jnp.pi/4) < atol and coords[1] < atol and coords[2] < atol:
        return 'cnot'
        
    # SWAP: (pi/4, pi/4, pi/4)
    if jnp.all(jnp.abs(coords - jnp.pi/4) < atol):
        return 'swap'
        
    # iSWAP: (pi/4, pi/4, 0)
    if jnp.abs(coords[0] - jnp.pi/4) < atol and jnp.abs(coords[1] - jnp.pi/4) < atol and coords[2] < atol:
        return 'iswap'
        
    return 'generic'

def get_sycamore_initial_params(coords):
    """
    Returns initial parameters for the Sycamore ansatz based on KAK coords.
    This serves as a seed for VQE or a direct solution for known gates.
    """
    # Params shape: (4, 2, 3)
    
    # Hardcoded CNOT params (found via optimization)
    cnot_params = jnp.array([
        [[1.5007599592208862, 2.0785629749298096, -0.4461745321750641], 
         [-0.17979516088962555, 0.09397346526384354, -0.40404394268989563]], 
        [[-0.37198543548583984, 0.4811127781867981, 0.6105138063430786], 
         [-1.5597529411315918, 2.2263600826263428, -1.9548892974853516]], 
        [[0.7979231476783752, -0.020915437489748, 1.1834030151367188], 
         [1.469095230102539, 0.7153650522232056, 0.46262872219085693]], 
        [[0.19084732234477997, -1.5133144855499268, -1.2263277769088745], 
         [1.997868537902832, -0.07750305533409119, 0.9836772680282593]]
    ], dtype=jnp.float32)

    # Simple logic:
    # If small coords (Identity-like) -> zeros
    # Else -> CNOT params (as a generic good seed)
    is_small = jnp.all(coords < 0.1)
    
    return jnp.where(is_small, jnp.zeros((4, 2, 3), dtype=jnp.float32), cnot_params)
