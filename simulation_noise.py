import cirq
import numpy as np

def create_non_trivial_circuit(qubits, depth=5):
    """Creates a random circuit with entanglement."""
    circuit = cirq.Circuit()
    
    # Initial Hadamard layer
    circuit.append(cirq.H.on_each(qubits))
    
    for _ in range(depth):
        # Random single qubit gates
        for q in qubits:
            circuit.append(cirq.rz(np.random.uniform(0, 2 * np.pi)).on(q))
            circuit.append(cirq.rx(np.random.uniform(0, 2 * np.pi)).on(q))
            
        # Entangling layer (CZ gates between adjacent qubits if GridQubits)
        # For simplicity in this script, we'll just chain them if they are in a list, 
        # or use a specific pattern for GridQubits.
        if isinstance(qubits[0], cirq.GridQubit):
            # Apply CZ to adjacent pairs
            for i in range(len(qubits)):
                for j in range(i + 1, len(qubits)):
                    q1 = qubits[i]
                    q2 = qubits[j]
                    if q1.is_adjacent(q2):
                        # 50% chance to apply gate to reduce depth/complexity if needed
                        if np.random.random() > 0.5:
                            circuit.append(cirq.CZ(q1, q2))
        else:
             # Fallback for LineQubits or others
            for i in range(0, len(qubits) - 1, 2):
                circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
                
    return circuit

def add_noise_to_circuit(circuit, noise_prob=0.01):
    """Adds depolarizing noise to the circuit."""
    noisy_circuit = cirq.Circuit()
    for moment in circuit:
        noisy_circuit.append(moment)
        # Apply noise after each moment
        # We apply noise to all qubits active in this moment, or all qubits in general?
        # A common simple model is noise on all qubits after every time step (idle noise + gate noise)
        # But here we'll just apply it to the qubits operated on, or all for simplicity.
        # Let's apply to all qubits in the circuit for a "global background noise" feel,
        # or just after gates. Let's do after gates for now.
        for op in moment:
            for q in op.qubits:
                noisy_circuit.append(cirq.depolarize(p=noise_prob).on(q))
    return noisy_circuit

def main():
    # 1. Setup Qubits (> 20)
    # 3x7 Grid = 21 qubits (Satisfies > 20 condition while keeping simulation fast)
    rows = 3
    cols = 7
    qubits = [cirq.GridQubit(r, c) for r in range(rows) for c in range(cols)]
    print(f"Created {len(qubits)} qubits in a {rows}x{cols} grid.")

    # 2. Create Circuit
    print("Generating circuit...")
    original_circuit = create_non_trivial_circuit(qubits, depth=5)
    
    # 3. Apply Noise
    print("Applying noise model...")
    noise_level = 0.01 # 1% error rate
    noisy_circuit = original_circuit.with_noise(cirq.depolarize(p=noise_level))
    
    # Add measurements at the end
    noisy_circuit.append(cirq.measure(*qubits, key='result'))

    # 4. Simulate
    print("Running simulation (this may take a moment)...")
    simulator = cirq.Simulator()
    
    # Run a few repetitions
    repetitions = 100
    result = simulator.run(noisy_circuit, repetitions=repetitions)

    print("\nSimulation complete.")
    print(f"Ran {repetitions} shots.")
    
    # Process results
    # Just show the counts of the first few measured states to verify output
    # Since 25 qubits = 2^25 states, we won't print the histogram.
    measurements = result.measurements['result']
    print(f"Measurement shape: {measurements.shape}")
    print(f"First 5 measurements:\n{measurements[:5]}")
    
    # Calculate expectation value of Z on the first qubit just to show some analysis
    # Map 0 -> 1 and 1 -> -1
    z_vals = [1 if m[0] == 0 else -1 for m in measurements]
    print(f"\n<Z> on Qubit(0,0): {np.mean(z_vals)}")

if __name__ == "__main__":
    main()
