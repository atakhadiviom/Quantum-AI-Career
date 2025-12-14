
import cirq
import cirq_google
import numpy as np

def analyze_cnot_decomposition():
    print("Analyzing CNOT Decomposition into Sycamore...")
    q0, q1 = cirq.LineQubit.range(2)
    cnot = cirq.CNOT(q0, q1)
    
    gateset = cirq_google.SycamoreTargetGateset()
    result = gateset.decompose_to_target_gateset(cnot, 0)
    
    # Flatten
    flat_ops = []
    if isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, cirq.Moment):
                flat_ops.extend(item.operations)
            else:
                flat_ops.append(item)
    else:
        flat_ops = [result]
        
    print(f"\nDecomposed CNOT into {len(flat_ops)} operations:")
    
    syc_type = type(cirq_google.SYC)
    
    layer_idx = 0
    
    # We want to map this to our format: 4 layers of single qubit gates (PhasedXZ), interleaved with Sycamore.
    # The format I established in 'worker_pure_jax_pipeline' return values is:
    # (4, 2, 3) -> 4 layers of PhasedXZ (Pre-SYC1, Mid-SYC1-2, Mid-SYC2-3, Post-SYC3?)
    # Wait, 2 SYC gates means: K1 - SYC - K2 - SYC - K3
    
    for op in flat_ops:
        if isinstance(op.gate, syc_type):
            print(f"  -- SYCAMORE --")
        elif isinstance(op.gate, cirq.PhasedXZGate):
            print(f"  PhasedXZ(q={op.qubits[0]}): x={op.gate.x_exponent}, z={op.gate.z_exponent}, a={op.gate.axis_phase_exponent}")
        else:
            print(f"  {op}")
            
    # Verify unitary
    circuit = cirq.Circuit(flat_ops)
    u_decomp = circuit.unitary()
    u_cnot = cirq.unitary(cnot)
    
    overlap = abs(np.trace(u_cnot.conj().T @ u_decomp)) / 4.0
    print(f"\nFidelity: {overlap}")

if __name__ == "__main__":
    analyze_cnot_decomposition()
