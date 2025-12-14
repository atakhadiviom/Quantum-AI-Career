
import cirq
import cirq_google
import numpy as np

def get_cnot_params():
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

    # We expect structure:
    # K1 (Layer 0)
    # SYC
    # K2 (Layer 1)
    # SYC
    # K3 (Layer 2)
    
    # Let's group them by looking for SYC
    k_layers = [[], [], []]
    current_layer = 0
    syc_count = 0
    
    syc_type = type(cirq_google.SYC)
    
    for op in flat_ops:
        if isinstance(op.gate, syc_type):
            current_layer += 1
            syc_count += 1
        else:
            k_layers[current_layer].append(op)
            
    print(f"Found {syc_count} SYC gates. Layers: {[len(l) for l in k_layers]}")
    
    # Now merge each layer into a single PhasedXZ per qubit
    merged_params = []
    
    for i, layer_ops in enumerate(k_layers):
        print(f"Layer {i}:")
        layer_circuit = cirq.Circuit(layer_ops)
        
        # We need to extract the PhasedXZ parameters for q0 and q1
        # Strategy: Compute unitary of the layer for each qubit? 
        # No, the layer might be entangling if we messed up, but here they are local.
        # We assume local gates.
        
        # Extract local unitaries
        u_layer = layer_circuit.unitary()
        
        # Check if diagonal (tensor product)
        # Actually, simpler: decompose the layer's unitary into PhasedXZ
        # Using cirq.single_qubit_decompose_to_phased_xz
        
        # We need to isolate q0 and q1 parts.
        # Since they are local, we can just run the circuit on isolated qubits
        
        layer_params = []
        for q in [q0, q1]:
            # Filter ops for this qubit
            q_ops = [op for op in layer_ops if q in op.qubits]
            if not q_ops:
                # Identity
                layer_params.append((0, 0, 0))
                continue
                
            u_q = cirq.Circuit(q_ops).unitary()
            try:
                gate = cirq.PhasedXZGate.from_matrix(u_q)
            except:
                # Fallback if specific method doesn't exist (older cirq?)
                # Try geometric decomposition
                from cirq.linalg import decompose_one_qubit_unitary_into_interaction_coefficients
                # This is too low level.
                # Just assuming from_matrix works in 1.6.1
                gate = cirq.PhasedXZGate.from_matrix(u_q)
                
            print(f"  Q{q.x}: x={gate.x_exponent}, z={gate.z_exponent}, a={gate.axis_phase_exponent}")
            layer_params.append((gate.x_exponent, gate.z_exponent, gate.axis_phase_exponent))
        
        merged_params.append(layer_params)
        
    print("\nCONSTANTS FOR JAX:")
    print("params = jnp.array([")
    for layer in merged_params:
        print(f"    [{layer[0]}, {layer[1]}],")
    print("])")

if __name__ == "__main__":
    get_cnot_params()
