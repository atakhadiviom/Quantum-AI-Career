
import jax
import jax.numpy as jnp
import numpy as np

def debug_kak_pure_jax(name, u_np):
    print(f"--- {name} ---")
    u = jnp.array(u_np)
    
    # Constants
    inv_sqrt2 = 1.0 / jnp.sqrt(2.0)
    Q = jnp.array([
        [1, 0, 0, 1j],
        [0, 1j, 1, 0],
        [0, 1j, -1, 0],
        [1, 0, 0, -1j]
    ], dtype=jnp.complex64) * inv_sqrt2
    
    # Magic basis
    m = jnp.conj(Q).T @ u @ Q
    gamma = m.T @ m
    
    # Eigenvalues
    eigvals = jnp.linalg.eigvals(gamma)
    print(f"Eigvals: {eigvals}")
    
    # Angles
    angles = -jnp.angle(eigvals) / 2.0
    print(f"Angles: {angles}")
    
    # Try a different magic basis (Standard definition)
    # The one in code was:
    # Q = [
    #    [1, 0, 0, 1j],
    #    [0, 1j, 1, 0],
    #    [0, 1j, -1, 0],
    #    [1, 0, 0, -1j]
    # ]
    # This might be different from what gives canonical x,y,z directly.
    
    # Try computing trace
    # Tr(M) = sum(eigvals)
    # x,y,z are related to Tr(M)?
    
    # Let's look at the characteristic polynomial.
    # But we want JAX compatibility (no roots finding if possible, eigvals is fine).
    
    # Let's try to infer x,y,z from sorted angles.
    # CNOT Angles: [0, 0, -pi/2, -pi/2]
    # Diff: pi/2. Half diff: pi/4. Matches x=pi/4.
    
    # SWAP Angles: [0, 0, 0, 0]
    # This is weird. SWAP should be (pi/4, pi/4, pi/4).
    # Maybe SWAP needs a global phase adjustment?
    # SWAP = exp(i * pi/4 * (XX + YY + ZZ)) * phase?
    # SWAP maps |00>->|00>, |01>->|10>, |10>->|01>, |11>->|11>
    # Magic basis transform of SWAP:
    # M_swap = diag(1, 1, 1, -1) ? No.
    
    # Calculate spread and mean for heuristic tuning
    a = jnp.mod(angles, jnp.pi)
    v = jnp.sort(a)
    spread = v[3] - v[0]
    mean = jnp.mean(v)
    
    print(f"Spread: {spread}")
    print(f"Mean: {mean}")
    
    pass# U_magic for SWAP has -1 on diagonal at index 2 (0-indexed)?
    # Diagonal: 1, 1, -1, 1.
    # The M matrix is U_magic.T @ U_magic.
    # Since U_magic is diagonal and real, M = U_magic^2.
    # 1^2 = 1. (-1)^2 = 1.
    # So M = I. Eigenvalues are all 1. Angles are 0.
    
    # This means calculating M = U_m^T U_m destroys the sign information if U_m is real.
    # But U_m is complex in general.
    # However, for SWAP, U_m is real.
    
    # We need to extract the angles from U_magic directly, not M, if we want to distinguish.
    # KAK decomposition: U = k1 exp(i sum ...) k2.
    # In magic basis: U_magic = k1_m D k2_m.
    # If we assume k1, k2 are local gates (product of SU(2)), then k1_m, k2_m are orthogonal?
    
    # Standard KAK algorithm involves M = U_m^T U_m.
    # Eigenvalues of M are exp(2i h_j).
    # If h_j = pi/4, then 2 h_j = pi/2. exp(i pi/2) = i.
    # If h_j = -pi/4, exp(-i pi/2) = -i.
    
    # For SWAP (pi/4, pi/4, pi/4):
    # h = (pi/4, pi/4, pi/4).
    # M eigenvalues should be exp(2i pi/4) = i?
    # Wait, my SWAP calculation gave 1.
    
    # Maybe Q is wrong?
    # Cirq's Q:
    # [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]] * 1/sqrt(2)
    # This matches my code.
    
    # Let's check Cirq's implementation of SWAP interaction coeffs.
    # Cirq says (pi/4, pi/4, pi/4).
    
    # Let's try to adjust SWAP by a global phase to see if it fixes M.
    # SWAP has determinant -1? No, 1.
    # SWAP = [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]. Det = -1?
    # Row swaps: 1 (row 1 <-> 2).
    # Det is -1.
    # SU(4) requires Det = 1.
    # So SWAP is not in SU(4).
    # iSWAP is in SU(4)?
    # iSWAP = [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]. Det = i*i - 0 = -1?
    
    # We must project to SU(4) first.
    # Det(U) = d.
    # U_su4 = U / d^(1/4).
    
    # Let's calculate Det(U) and normalize.
    det = jnp.linalg.det(u)
    print(f"Det: {det}")
    u_su4 = u / (det ** 0.25)
    
    # Re-calculate M with U_su4
    m_su4 = jnp.conj(Q).T @ u_su4 @ Q
    gamma_su4 = m_su4.T @ m_su4
    eigvals_su4 = jnp.linalg.eigvals(gamma_su4)
    print(f"Eigvals SU4: {eigvals_su4}")
    angles_su4 = -jnp.angle(eigvals_su4) / 2.0
    print(f"Angles SU4: {angles_su4}")
    
    pass

cnot = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=np.complex64)

swap = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=np.complex64)

debug_kak_pure_jax("CNOT", cnot)
debug_kak_pure_jax("SWAP", swap)
