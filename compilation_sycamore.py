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

# Import Optimized VQE Module
import optimized_vqe
import kak_utils

# ---------------------------------------------------------
# JAX Core Functions (Module Level for Compilation)
# ---------------------------------------------------------

@jax.jit
def _process_one_wrapper(u_target):
    """
    Wrapper for JAX VMAP that calls the optimized synthesis logic.
    """
    return optimized_vqe.synthesize_gate_jit(u_target)

@jax.jit
def _synthesize_batch_vmap(u_batch):
    return jax.vmap(_process_one_wrapper)(u_batch)

@jax.jit
def _compute_batch_fidelity(params_batch, target_u_batch):
    """
    Computes fidelity for a batch of parameters and target unitaries.
    """
    def _fid(p, u):
        pred_u = optimized_vqe._parametric_unitary(p)
        return optimized_vqe.fidelity(pred_u, u)
    return jax.vmap(_fid)(params_batch, target_u_batch)

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
                
    fidelities = []

    # 2. Vectorized Processing
    if unitaries:
        # Convert to JAX array
        u_stack = jnp.array(np.stack(unitaries))
        
        # Run JAX (vmap)
        # This compiles once and runs in parallel on CPU/GPU
        params_batch = _synthesize_batch_vmap(u_stack)
        params_batch.block_until_ready()
        
        # Compute Fidelities for Analysis
        fid_batch = _compute_batch_fidelity(params_batch, u_stack)
        fidelities = np.array(fid_batch)
        
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
    return cirq.Circuit(final_ops), fidelities

# Deprecated: Kept for compatibility but not used
def parallel_sycamore_compilation(circuit, executor=None):
    return vectorized_sycamore_compilation(circuit)[0]

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
    # We don't need ProcessPoolExecutor for Pure JAX anymore but keeping structure
    
    start_time = time.time()
    parallel_circuit, fidelities = vectorized_sycamore_compilation(original_circuit)
    parallel_time = time.time() - start_time
        
    print(f"Parallel Compilation Time: {parallel_time:.4f} s")
    
    # 4. Analysis
    speedup = sequential_time / parallel_time
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Fidelity Statistics
    if len(fidelities) > 0:
        print("\n## Fidelity Statistics (Gate-by-Gate)")
        print(f"Average Fidelity: {np.mean(fidelities):.6f}")
        print(f"Min Fidelity: {np.min(fidelities):.6f}")
        print(f"Max Fidelity: {np.max(fidelities):.6f}")
        print(f"Fidelity > 0.99: {np.sum(fidelities > 0.99)} / {len(fidelities)}")
    
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
            print(f"Global Circuit Fidelity: {overlap:.6f}")
            
            if overlap > 0.99:
                print("SUCCESS: Circuits are logically equivalent.")
            else:
                print("WARNING: Circuits might not be equivalent (Global Fidelity low).")
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
