
import jax
import jax.numpy as jnp
import numpy as np
import kak_utils

# --- Ansatz from compilation_sycamore.py ---

def _phased_xz(x, z, a):
    # Constants
    I = jnp.eye(2, dtype=jnp.complex64)
    X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
    
    # Z^(a+z)
    exp_z = jnp.exp(-1j * jnp.pi * (a + z) / 2.0)
    mat_z = jnp.array([[exp_z, 0], [0, jnp.conj(exp_z)]], dtype=jnp.complex64)
    
    # X^x
    c = jnp.cos(jnp.pi * x / 2.0)
    s = jnp.sin(jnp.pi * x / 2.0)
    mat_x = c * I - 1j * s * X
    
    # Z^-a
    exp_ma = jnp.exp(-1j * jnp.pi * (-a) / 2.0)
    mat_ma = jnp.array([[exp_ma, 0], [0, jnp.conj(exp_ma)]], dtype=jnp.complex64)
    
    return mat_ma @ mat_x @ mat_z

def _parametric_unitary(p):
    # SYC Matrix
    syc_mat = jnp.array([
        [1, 0, 0, 0],
        [0, 0, -1j, 0],
        [0, -1j, 0, 0],
        [0, 0, 0, jnp.exp(-1j * jnp.pi / 6)]
    ], dtype=jnp.complex64)
    
    # Layer 0 (K1)
    u0_0 = _phased_xz(p[0,0,0], p[0,0,1], p[0,0,2])
    u0_1 = _phased_xz(p[0,1,0], p[0,1,1], p[0,1,2])
    u0 = jnp.kron(u0_0, u0_1)
    
    # Layer 1 (K2)
    u1_0 = _phased_xz(p[1,0,0], p[1,0,1], p[1,0,2])
    u1_1 = _phased_xz(p[1,1,0], p[1,1,1], p[1,1,2])
    u1 = jnp.kron(u1_0, u1_1)
    
    # Layer 2 (K3)
    u2_0 = _phased_xz(p[2,0,0], p[2,0,1], p[2,0,2])
    u2_1 = _phased_xz(p[2,1,0], p[2,1,1], p[2,1,2])
    u2 = jnp.kron(u2_0, u2_1)
    
    # Layer 3 (K4)
    u3_0 = _phased_xz(p[3,0,0], p[3,0,1], p[3,0,2])
    u3_1 = _phased_xz(p[3,1,0], p[3,1,1], p[3,1,2])
    u3 = jnp.kron(u3_0, u3_1)
    
    # Circuit: K1 -> SYC -> K2 -> SYC -> K3 -> SYC -> K4
    # Matmul order: U_total = K4 @ SYC @ K3 @ SYC @ K2 @ SYC @ K1
    
    return u3 @ syc_mat @ u2 @ syc_mat @ u1 @ syc_mat @ u0

def _loss_fn(p, target_u):
    predicted_u = _parametric_unitary(p)
    overlap = jnp.abs(jnp.trace(jnp.conj(target_u).T @ predicted_u)) / 4.0
    return 1.0 - overlap

def optimize_single(target_u, seed_idx):
    # Initialize
    rng_key = jax.random.PRNGKey(seed_idx)
    
    # Try KAK seed for idx 0, random for others
    if seed_idx == 0:
        kak_coords = kak_utils.compute_kak_coords(target_u)
        init_params = kak_utils.get_sycamore_initial_params(kak_coords)
        print("Using KAK Initialization")
    else:
        init_params = jax.random.normal(rng_key, (4, 2, 3)) * jnp.pi
        print("Using Random Initialization")
        
    lr = 0.05
    steps = 500
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    
    m = jnp.zeros_like(init_params)
    v = jnp.zeros_like(init_params)
    
    params = init_params
    
    print(f"Start Loss: {_loss_fn(params, target_u)}")
    
    for t in range(1, steps + 1):
        loss, grads = jax.value_and_grad(_loss_fn)(params, target_u)
        
        # Adam
        m = b1 * m + (1 - b1) * grads
        v = b2 * v + (1 - b2) * (grads ** 2)
        
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        
        params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        
        if t % 50 == 0:
            print(f"Step {t}: Loss {loss:.6f}")
            
    return params, loss

import cirq
import cirq_google

def check_syc_consistency():
    print("\nChecking SYC Consistency...")
    syc_cirq = cirq.unitary(cirq_google.SYC)
    
    syc_jax = jnp.array([
        [1, 0, 0, 0],
        [0, 0, -1j, 0],
        [0, -1j, 0, 0],
        [0, 0, 0, jnp.exp(-1j * jnp.pi / 6)]
    ], dtype=jnp.complex64)
    
    # Check difference
    diff = np.abs(syc_cirq - syc_jax)
    print(f"Max Diff SYC: {np.max(diff)}")

def check_phxz_consistency():
    print("\nChecking PhasedXZ Consistency...")
    # Test random parameters
    key = jax.random.PRNGKey(0)
    for i in range(5):
        params = jax.random.normal(key, (3,)) + i
        x, z, a = params[0], params[1], params[2]
        # Wrap to reasonable range? No, formulas are periodic
        
        # JAX
        u_jax = _phased_xz(x, z, a)
        
        # Cirq
        gate = cirq.PhasedXZGate(x_exponent=float(x), z_exponent=float(z), axis_phase_exponent=float(a))
        u_cirq = cirq.unitary(gate)
        
        # Check trace overlap (ignoring global phase)
        overlap = jnp.abs(jnp.trace(jnp.conj(u_cirq).T @ u_jax)) / 2.0
        print(f"Params (x={x:.2f}, z={z:.2f}, a={a:.2f}) -> Overlap: {overlap:.6f}")
        
        if overlap < 0.99:
             print("MISMATCH DETECTED!")
             # Debug matrices
             # print("JAX:\n", u_jax)
             # print("Cirq:\n", u_cirq)

def reconstruct_and_verify(params, target_u):
    print("\nReconstructing and Verifying with Cirq...")
    q0, q1 = cirq.LineQubit.range(2)
    ops = []
    p = params
    
    # Layer 0 (K1)
    ops.append(cirq.PhasedXZGate(x_exponent=p[0,0,0], z_exponent=p[0,0,1], axis_phase_exponent=p[0,0,2])(q0))
    ops.append(cirq.PhasedXZGate(x_exponent=p[0,1,0], z_exponent=p[0,1,1], axis_phase_exponent=p[0,1,2])(q1))
    
    # SYC 1
    ops.append(cirq_google.SYC(q0, q1))
    
    # Layer 1 (K2)
    ops.append(cirq.PhasedXZGate(x_exponent=p[1,0,0], z_exponent=p[1,0,1], axis_phase_exponent=p[1,0,2])(q0))
    ops.append(cirq.PhasedXZGate(x_exponent=p[1,1,0], z_exponent=p[1,1,1], axis_phase_exponent=p[1,1,2])(q1))
    
    # SYC 2
    ops.append(cirq_google.SYC(q0, q1))
    
    # Layer 2 (K3)
    ops.append(cirq.PhasedXZGate(x_exponent=p[2,0,0], z_exponent=p[2,0,1], axis_phase_exponent=p[2,0,2])(q0))
    ops.append(cirq.PhasedXZGate(x_exponent=p[2,1,0], z_exponent=p[2,1,1], axis_phase_exponent=p[2,1,2])(q1))
    
    # SYC 3
    ops.append(cirq_google.SYC(q0, q1))
    
    # Layer 3 (K4)
    ops.append(cirq.PhasedXZGate(x_exponent=p[3,0,0], z_exponent=p[3,0,1], axis_phase_exponent=p[3,0,2])(q0))
    ops.append(cirq.PhasedXZGate(x_exponent=p[3,1,0], z_exponent=p[3,1,1], axis_phase_exponent=p[3,1,2])(q1))
    
    circuit = cirq.Circuit(ops)
    u_cirq = cirq.unitary(circuit)
    
    overlap = np.abs(np.trace(np.conj(target_u).T @ u_cirq)) / 4.0
    print(f"Reconstructed Fidelity: {overlap}")

def main():
    check_syc_consistency()
    check_phxz_consistency()
    
    print("Generating random unitary...")
    # Random unitary via QR
    m = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    q, r = np.linalg.qr(m)
    target_u = jnp.array(q)
    
    print("Optimizing...")
    # Try KAK seed
    params_k, loss_k = optimize_single(target_u, 0)
    print(f"Final KAK Loss: {loss_k}")
    
    # Try Random seed
    params_r, loss_r = optimize_single(target_u, 42)
    print(f"Final Random Loss: {loss_r}")
    
    reconstruct_and_verify(params_r, target_u)

if __name__ == "__main__":
    main()
