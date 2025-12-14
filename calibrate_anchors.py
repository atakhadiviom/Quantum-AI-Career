
import jax
import jax.numpy as jnp
import numpy as np
import kak_utils

# --- Ansatz from compilation_sycamore.py (CORRECTED) ---

@jax.jit
def _phased_xz(x, z, a):
    # U = Z^(z+a) X^x Z^-a (derived from Cirq PhasedXZGate definition)
    # Constants
    I = jnp.eye(2, dtype=jnp.complex64)
    X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    
    # Right: Z^-a
    exp_ma = jnp.exp(-1j * jnp.pi * (-a) / 2.0)
    mat_right = jnp.array([[exp_ma, 0], [0, jnp.conj(exp_ma)]], dtype=jnp.complex64)
    
    # Middle: X^x
    c = jnp.cos(jnp.pi * x / 2.0)
    s = jnp.sin(jnp.pi * x / 2.0)
    mat_x = c * I - 1j * s * X
    
    # Left: Z^(z+a)
    exp_za = jnp.exp(-1j * jnp.pi * (z + a) / 2.0)
    mat_left = jnp.array([[exp_za, 0], [0, jnp.conj(exp_za)]], dtype=jnp.complex64)
    
    return mat_left @ mat_x @ mat_right

# Helper for SYC Unitary (Fixed)
syc_mat = jnp.array([
    [1, 0, 0, 0],
    [0, 0, -1j, 0],
    [0, -1j, 0, 0],
    [0, 0, 0, jnp.exp(-1j * jnp.pi / 6)]
], dtype=jnp.complex64)

@jax.jit
def _parametric_unitary(p):
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
    
    return u3 @ syc_mat @ u2 @ syc_mat @ u1 @ syc_mat @ u0

@jax.jit
def _loss_fn(p, target_u):
    predicted_u = _parametric_unitary(p)
    overlap = jnp.abs(jnp.trace(jnp.conj(target_u).T @ predicted_u)) / 4.0
    return 1.0 - overlap

def find_cnot_params():
    print("Finding CNOT params...")
    # CNOT Unitary
    cnot = jnp.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=jnp.complex64)
    
    # Init with random
    key = jax.random.PRNGKey(0)
    
    # Optimize
    lr = 0.05
    steps = 2000
    b1, b2, eps = 0.9, 0.999, 1e-8
    
    @jax.jit
    def update(params, m, v, t):
        loss, grads = jax.value_and_grad(_loss_fn)(params, cnot)
        m = b1 * m + (1 - b1) * grads
        v = b2 * v + (1 - b2) * (grads ** 2)
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        new_params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        return new_params, m, v, loss

    # Try a few seeds
    best_loss = 1.0
    best_params = None
    
    for i in range(10):
        params = jax.random.normal(jax.random.PRNGKey(i), (4, 2, 3)) * jnp.pi
        m = jnp.zeros_like(params)
        v = jnp.zeros_like(params)
        
        for t in range(1, steps + 1):
            params, m, v, loss = update(params, m, v, t)
            
        print(f"Seed {i}: Loss {loss}")
        if loss < best_loss:
            best_loss = loss
            best_params = params
            
    print(f"Best CNOT Loss: {best_loss}")
    # print(f"Best CNOT Params:\n{best_params}")
    return best_params

def main():
    p_cnot = find_cnot_params()
    
    # Format for copy-paste
    print("\n# Copy this into kak_utils.py:")
    print("    params_cnot = jnp.array([")
    for i in range(4):
        print(f"        {p_cnot[i].tolist()},")
    print("    ])")

if __name__ == "__main__":
    main()
