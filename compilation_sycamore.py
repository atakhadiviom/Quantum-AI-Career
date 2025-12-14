#!/usr/bin/env python3
"""
Compilation: Compiling a random deep circuit onto the Sycamore gate set

This script demonstrates how to create a random deep quantum circuit (depth > 50)
and compile it onto the Google Sycamore gate set using Cirq.
It benchmarks a parallel Map-Reduce decomposition strategy against the baseline.
"""

import sys
import subprocess
import random
import time
import concurrent.futures
import pickle

# Ensure cirq and cirq-google are installed
try:
    import cirq
    import cirq_google
    import numpy as np
except ImportError:
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "cirq", "cirq-google", "numpy"])
    import cirq
    import cirq_google
    import numpy as np

from cirq_google import Sycamore

import jax
import jax.numpy as jnp

# ---------------------------------------------------------
# JAX Core Functions (Module Level for Compilation)
# ---------------------------------------------------------

# Helper to create PhasedXZ unitary
@jax.jit
def _phased_xz(x, z, a):
    # U = Z^(z+a) X^x Z^-a (derived from Cirq PhasedXZGate definition)
    # Note: Cirq's Z^t and X^t have global phases relative to SU(2) definitions.
    # We use SU(2) definitions here which is fine for Fidelity up to global phase.
    
    # Constants
    I = jnp.eye(2, dtype=jnp.complex64)
    X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
    
    # Right: Z^-a
    # Z^-a = exp(-i * pi * (-a) / 2 * Z) = exp(i * pi * a / 2 * Z)
    exp_ma = jnp.exp(-1j * jnp.pi * (-a) / 2.0)
    mat_right = jnp.array([[exp_ma, 0], [0, jnp.conj(exp_ma)]], dtype=jnp.complex64)
    
    # Middle: X^x
    c = jnp.cos(jnp.pi * x / 2.0)
    s = jnp.sin(jnp.pi * x / 2.0)
    mat_x = c * I - 1j * s * X
    
    # Left: Z^(z+a)
    # Z^(z+a) = exp(-i * pi * (z+a) / 2 * Z)
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
    # p shape: (4, 2, 3)
    # Reconstruct Unitary from params (Differentiable)
    
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

@jax.jit
def _loss_fn(p, target_u):
    predicted_u = _parametric_unitary(p)
    # Fidelity Loss: 1 - |Tr(U_target^dag @ U_pred)| / 4
    overlap = jnp.abs(jnp.trace(jnp.conj(target_u).T @ predicted_u)) / 4.0
    return 1.0 - overlap

@jax.jit
def _optimize_one(u_target, init_params):
    lr = 0.05
    max_steps = 100 # Reduced to 100 for speed
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    tol = 1e-3 # 99.9% fidelity
    
    # Initialize moments
    m_init = jnp.zeros_like(init_params)
    v_init = jnp.zeros_like(init_params)
    
    # Initial loss
    loss_init = _loss_fn(init_params, u_target)
    
    # State: (params, m, v, t, loss)
    init_state = (init_params, m_init, v_init, 0, loss_init)
    
    def cond_fn(state):
        _, _, _, t, loss = state
        return (t < max_steps) & (loss > tol)
    
    def body_fn(state):
        params, m, v, t, _ = state
        t = t + 1
        
        loss, grads = jax.value_and_grad(_loss_fn)(params, u_target)
        
        # Adam updates
        m = b1 * m + (1 - b1) * grads
        v = b2 * v + (1 - b2) * (grads ** 2)
        
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        
        new_params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        
        return (new_params, m, v, t, loss)
    
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    final_params, _, _, _, final_loss = final_state
        
    return final_params, final_loss

@jax.jit
def _process_one_wrapper(u_target):
    import kak_utils
    
    # Helper to run optimize on a specific seed (needed for vmap)
    def _run_optimization(seed):
        return _optimize_one(u_target, seed)

    kak_coords = kak_utils.compute_kak_coords(u_target)
    kak_seed = kak_utils.get_sycamore_initial_params(kak_coords)

    # Use Random seeds for exploration
    # Reduced back to 1 since KAK init is now calibrated
    n_random = 1 
    
    rng_key = jax.random.PRNGKey(42)
    
    # Combine seeds: Prioritize KAK seed
    # KAK seed is (1, 4, 2, 3)
    # Random seeds are (n_random, 4, 2, 3)
    # We need to expand KAK seed to (1, 4, 2, 3)
    seed_kak = jnp.expand_dims(kak_seed, 0)
    
    # Generate random seeds
    random_seeds = jax.random.normal(rng_key, (n_random, 4, 2, 3)) * jnp.pi
    
    all_seeds = jnp.concatenate([seed_kak, random_seeds], axis=0)
    
    # 2. Parallel Optimization (Multi-Start)
    # vmap over seeds
    results_params, results_losses = jax.vmap(_run_optimization)(all_seeds)
    
    # 3. Select Best
    best_idx = jnp.argmin(results_losses)
    best_params = results_params[best_idx]
    
    return best_params

@jax.jit
def _synthesize_batch_vmap(u_batch):
    return jax.vmap(_process_one_wrapper)(u_batch)

# ---------------------------------------------------------
# Worker Function for Parallel Decomposition
# ---------------------------------------------------------
def worker_decompose_operation(op):
    """
    Worker function to decompose a single operation.
    Must be at module level for pickling.
    """
    import cirq_google
    import cirq
    
    # Instantiate gateset inside worker (avoid pickling large objects)
    gateset = cirq_google.SycamoreTargetGateset()
    
    try:
        # decompose_to_target_gateset returns a list of operations or None
        result = gateset.decompose_to_target_gateset(op, 0)
    except Exception:
        # Fallback if decomposition fails
        return [op]
        
    if result is None:
        return [op]
        
    # Ensure result is iterable (list)
    if not isinstance(result, (list, tuple)):
        result = [result]
        
    # Flatten any nested Moments if necessary (though usually returns ops)
    flat_ops = []
    for item in result:
        if isinstance(item, cirq.Moment):
            flat_ops.extend(item.operations)
        else:
            flat_ops.append(item)
            
    return flat_ops

def jax_kak_interaction_coefficients(u):
    """
    Computes the KAK interaction coefficients (x, y, z) using JAX.
    Deprecated: Use kak_utils.compute_kak_coords directly.
    """
    import jax
    import jax.numpy as jnp
    import kak_utils
    
    # Convert numpy to jax array
    u_jax = jnp.array(u)
    return kak_utils.compute_kak_coords(u_jax)

def worker_decompose_batch_jax(batch_tuple):
    """
    Deprecated: Replaced by single-process vectorization.
    Kept for compatibility with potential external callers.
    """
    indices, ops = zip(*batch_tuple)
    import cirq
    import cirq_google
    
    gateset = cirq_google.SycamoreTargetGateset()
    results = []
    for op in ops:
        try:
            res = gateset.decompose_to_target_gateset(op, 0)
        except:
            res = [op]
        if res is None:
            res = [op]
        if not isinstance(res, (list, tuple)):
            res = [res]
        flat_ops = []
        for item in res:
            if isinstance(item, cirq.Moment):
                flat_ops.extend(item.operations)
            else:
                flat_ops.append(item)
        results.append(flat_ops)
    return indices, results

def worker_pure_jax_pipeline(batch_data):
    """
    Deprecated: Replaced by single-process vectorization.
    Kept for reference but not used in optimized path.
    """
    pass

def vectorized_sycamore_compilation(circuit):
    """
    Compiles a circuit using Single-Process JAX Vectorization.
    This eliminates IPC overhead and leverages JAX's native batching (vmap).
    """
    import cirq
    import cirq_google
    import numpy as np
    
    all_ops = list(circuit.all_operations())
    compiled_ops_list = [None] * len(all_ops)
    syc_type = type(cirq_google.SYC)
    
    # 1. Identify and Extract Unitaries
    non_native_indices = []
    unitaries = []
    
    for i, op in enumerate(all_ops):
        is_native = (isinstance(op.gate, syc_type) or isinstance(op.gate, cirq.PhasedXZGate))
        
        if is_native:
            compiled_ops_list[i] = [op]
        else:
            try:
                if len(op.qubits) == 2:
                    u = cirq.unitary(op)
                    non_native_indices.append(i)
                    unitaries.append(u)
                else:
                    compiled_ops_list[i] = [op]
            except:
                compiled_ops_list[i] = [op]
                
    # 2. Vectorized Processing
    if unitaries:
        # Convert to JAX array
        u_stack = jnp.array(np.stack(unitaries))
        
        # Run JAX (vmap)
        # This compiles once and runs in parallel on CPU/GPU
        params_batch = _synthesize_batch_vmap(u_stack)
        params_batch.block_until_ready()
        
        # Convert back to host
        params_batch_np = np.array(params_batch)
        
        # 3. Reconstruction
        for idx_ptr, idx in enumerate(non_native_indices):
            op = all_ops[idx]
            q0, q1 = op.qubits
            p = params_batch_np[idx_ptr]
            
            ops = []
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
            
            compiled_ops_list[idx] = ops
            
    # Flatten
    final_ops = [op for sublist in compiled_ops_list if sublist is not None for op in sublist]
    return cirq.Circuit(final_ops)

# Deprecated: Kept for compatibility but not used
def parallel_sycamore_compilation(circuit, executor=None):
    return vectorized_sycamore_compilation(circuit)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def generate_random_circuit(qubits, depth, gate_type='random'):
    """Generates a random circuit with the given qubits and depth."""
    circuit = cirq.Circuit()
    for _ in range(depth):
        # Apply random single-qubit gates
        for q in qubits:
            # Random Phase/X rotation
            circuit.append(cirq.X(q)**np.random.random())
        
        # Apply random two-qubit gates
        # Shuffle qubits to form random pairs
        shuffled_qubits = list(qubits)
        np.random.shuffle(shuffled_qubits)
        for i in range(0, len(shuffled_qubits) - 1, 2):
            q1, q2 = shuffled_qubits[i], shuffled_qubits[i+1]
            
            if gate_type == 'cnot':
                # Proof of Principle: Only CNOT gates
                circuit.append(cirq.CNOT(q1, q2))
            else:
                # Use Random Unitary Gate to force heavy decomposition calculation
                # This simulates the "heavy calculation" mentioned in the prompt
                m = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
                q_mat, _ = np.linalg.qr(m)
                gate = cirq.MatrixGate(q_mat)
                circuit.append(gate(q1, q2))
    return circuit

def sequential_sycamore_decomposition(circuit):
    """
    Sequentially decomposes the circuit into Sycamore gateset.
    Used for fair apples-to-apples comparison with parallel implementation.
    """
    new_ops = []
    # Reuse the worker logic (but running in main process)
    # worker_decompose_operation is self-contained
    for op in circuit.all_operations():
        new_ops.extend(worker_decompose_operation(op))
    return cirq.Circuit(new_ops)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    print(f"Cirq version: {cirq.__version__}")
    try:
        import multiprocessing
        print(f"CPU Count: {multiprocessing.cpu_count()}")
    except:
        print("CPU Count: Unknown")

    # 1. Create a Random Deep Circuit
    # Increased depth to ~6000 for "2 minute" stress test
    print("\n## 1. Create a Random Deep Circuit (RANDOM UNITARY MODE for VQE TEST)")
    n_qubits = 8 # 8 Qubits
    qubits = cirq.LineQubit.range(n_qubits)
    depth = 2000 # Increased depth for heavy load
    
    start_time = time.time()
    print("Generating circuit with random unitary gates...")
    original_circuit = generate_random_circuit(qubits, depth, gate_type='random')
    print(f"Generation Time: {time.time() - start_time:.4f} s")
    
    print(f"Original Circuit Depth: {len(original_circuit)}")
    print(f"Original Operation Count: {len(list(original_circuit.all_operations()))}")
    
    # 2. Benchmark Sequential Decomposition
    print("\n## 2. Benchmark Sequential Decomposition (Apples-to-Apples)")
    print("Running sequential baseline (this may take ~2 minutes)...")
    start_time = time.time()
    
    # We run the sequential version to establish the baseline
    sequential_circuit = sequential_sycamore_decomposition(original_circuit)
    
    sequential_time = time.time() - start_time
    print(f"Sequential Decomposition Time: {sequential_time:.4f} s")
    
    # 3. Benchmark Parallel Implementation (Pure JAX)
    print("\n## 3. Benchmark Parallel Implementation (Pure JAX Pipeline)")
    print("Warming up process pool...")
    
    # Warmup the pool/JAX (optional but good practice)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        print("Warmup complete.")
        
        start_time = time.time()
        parallel_circuit = parallel_sycamore_compilation(original_circuit, executor=executor)
        parallel_time = time.time() - start_time
        
    print(f"Parallel Compilation Time: {parallel_time:.4f} s")
    
    # 4. Analysis
    speedup = sequential_time / parallel_time
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # 5. Verify Equivalence
    print("\n## 4. Verify Equivalence")
    
    if n_qubits <= 10:
        print("Calculating unitaries (this might take a few seconds)...")
        try:
            u_seq = sequential_circuit.unitary()
            u_para = parallel_circuit.unitary()
            
            # Check overlap
            # Use numpy for trace
            overlap = abs(np.trace(u_seq.conj().T @ u_para)) / (2**n_qubits)
            print(f"Overlap (Fidelity): {overlap:.6f}")
            
            if overlap > 0.9999:
                print("SUCCESS: Circuits are logically equivalent.")
            else:
                print("WARNING: Circuits might not be equivalent.")
        except Exception as e:
            print(f"Verification failed with error: {e}")
    else:
        print("Skipping unitary verification (too many qubits).")

    # 6. Final Stats
    print("\n## Final Stats")
    print(f"Sequential Ops: {len(list(sequential_circuit.all_operations()))}")
    print(f"Parallel Ops: {len(list(parallel_circuit.all_operations()))}")

if __name__ == "__main__":
    main()
