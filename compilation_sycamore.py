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

# ---------------------------------------------------------
# Worker Function for Parallel Decomposition
# ---------------------------------------------------------
def worker_decompose_operation(op):
    """
    Worker function to decompose a single operation into Sycamore gateset.
    Must be at module level for pickling.
    """
    # Re-import to ensure visibility in worker process
    import cirq_google
    import cirq
    
    # Instantiate the target gateset locally
    # (SycamoreTargetGateset is lightweight)
    gateset = cirq_google.SycamoreTargetGateset()
    
    # Decompose
    # decompose_to_target_gateset returns a list of operations/moments
    try:
        result = gateset.decompose_to_target_gateset(op, 0)
    except Exception:
        # Fallback if decomposition fails (should not happen for standard gates)
        return [op]
    
    # If result is None, it means it's already native
    if result is None:
        return [op]
        
    # Ensure result is iterable (list/tuple), if single op, wrap it
    if not isinstance(result, (list, tuple)):
        result = [result]
        
    # Flatten if it contains moments
    flat_ops = []
    for item in result:
        if isinstance(item, cirq.Moment):
            flat_ops.extend(item.operations)
        else:
            flat_ops.append(item)
    return flat_ops

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def generate_random_circuit(qubits, depth):
    """Generates a random circuit with the given qubits and depth."""
    circuit = cirq.Circuit()
    for _ in range(depth):
        # Apply random single-qubit gates
        for q in qubits:
            # Random Phase/X rotation
            circuit.append(cirq.X(q)**random.random())
        
        # Apply random two-qubit gates
        # Shuffle qubits to form random pairs
        shuffled_qubits = list(qubits)
        random.shuffle(shuffled_qubits)
        for i in range(0, len(shuffled_qubits) - 1, 2):
            q1, q2 = shuffled_qubits[i], shuffled_qubits[i+1]
            
            # Use Random Unitary Gate to force heavy decomposition calculation
            # This simulates the "heavy calculation" mentioned in the prompt
            m = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
            q_mat, _ = np.linalg.qr(m)
            gate = cirq.MatrixGate(q_mat)
            circuit.append(gate(q1, q2))
    return circuit

def count_sycamore_gates(circuit):
    """Counts the number of Sycamore (SYC) gates in the circuit."""
    count = 0
    syc_type = type(cirq_google.SYC)
    for op in circuit.all_operations():
        if isinstance(op.gate, syc_type):
            count += 1
    return count

def parallel_sycamore_compilation(circuit, executor=None):
    """
    Compiles the circuit using ProcessPoolExecutor for decomposition.
    """
    # 1. Identify Operations and Filter
    all_ops = list(circuit.all_operations())
    
    tasks = []     # List of ops to decompose
    indices = []   # Indices in the original list
    
    # Pre-fill with None
    compiled_ops_list = [None] * len(all_ops)
    
    syc_type = type(cirq_google.SYC)
    
    for i, op in enumerate(all_ops):
        # Check if already native
        # SycamoreTargetGateset native: SYC and PhasedXZ
        is_native = (isinstance(op.gate, syc_type) or 
                     isinstance(op.gate, cirq.PhasedXZGate))
        
        if is_native:
            compiled_ops_list[i] = [op]
        else:
            # Schedule for decomposition
            tasks.append(op)
            indices.append(i)
            
    # 2. Map Phase (Parallel)
    should_shutdown = False
    if executor is None:
        executor = concurrent.futures.ProcessPoolExecutor()
        should_shutdown = True
        
    try:
        # Determine chunksize for efficiency
        # Larger chunks are better to minimize IPC for uniform tasks
        workers = executor._max_workers if executor._max_workers else 1
        chunk_size = max(1, len(tasks) // workers)
        
        # map preserves order
        results = list(executor.map(worker_decompose_operation, tasks, chunksize=chunk_size))
    finally:
        if should_shutdown:
            executor.shutdown()
        
    # 3. Reduce Phase (Sequential Reconstruction)
    for i, result_ops in zip(indices, results):
        compiled_ops_list[i] = result_ops
        
    # Flatten the list of lists
    final_ops = [op for sublist in compiled_ops_list for op in sublist]
    
    # Reconstruct Circuit
    new_circuit = cirq.Circuit(final_ops)
    
    # Reconstruct Circuit
    new_circuit = cirq.Circuit(final_ops)
    
    return new_circuit

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
    print("\n## 1. Create a Random Deep Circuit")
    
    # Use parameters that create a heavy enough workload
    n_qubits = 10
    depth = 400 # Deep circuit with MatrixGates
    
    qubits = cirq.LineQubit.range(n_qubits)
    print("Generating circuit with random unitary gates...")
    original_circuit = generate_random_circuit(qubits, depth)
    
    print(f"Original Circuit Depth: {len(original_circuit)}")
    print(f"Original Operation Count: {len(list(original_circuit.all_operations()))}")

    # 2. Benchmark Sequential Decomposition
    print("\n## 2. Benchmark Sequential Decomposition (Apples-to-Apples)")
    
    start_time = time.perf_counter()
    sequential_circuit = sequential_sycamore_decomposition(original_circuit)
    sequential_time = time.perf_counter() - start_time
    print(f"Sequential Decomposition Time: {sequential_time:.4f} s")
    
    # 3. Benchmark Parallel Implementation
    print("\n## 3. Benchmark Parallel Implementation (Map-Reduce)")
    
    # Warmup the pool to simulate a long-running service / avoid startup cost
    print("Warming up process pool...")
    pool = concurrent.futures.ProcessPoolExecutor()
    # Submit dummy tasks to force worker creation and import loading
    ops = list(original_circuit.all_operations())
    dummy_tasks = ops[:min(16, len(ops))] 
    list(pool.map(worker_decompose_operation, dummy_tasks))
    print("Warmup complete.")
    
    start_time = time.perf_counter()
    parallel_circuit = parallel_sycamore_compilation(original_circuit, executor=pool)
    parallel_time = time.perf_counter() - start_time
    print(f"Parallel Compilation Time: {parallel_time:.4f} s")
    
    pool.shutdown()
    
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
