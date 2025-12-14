"""
Comprehensive test suite for quantum compilation framework.
Run with: pytest test_suite.py -v
"""

import pytest
import numpy as np
import jax.numpy as jnp
import cirq
import cirq_google
from typing import Callable

import kak_utils
from optimized_vqe import (
    create_parametric_unitary_fn,
    fidelity,
    synthesize_gate_smart
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def standard_gates():
    """Standard quantum gates for testing."""
    return {
        'I': np.eye(4, dtype=np.complex64),
        'CNOT': np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex64),
        'CZ': np.diag([1, 1, 1, -1]).astype(np.complex64),
        'SWAP': np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex64),
        'iSWAP': np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex64),
    }


@pytest.fixture
def random_unitaries():
    """Generate random unitaries for stress testing."""
    def _generate(n=10, seed=42):
        np.random.seed(seed)
        unitaries = []
        for _ in range(n):
            m = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
            q, r = np.linalg.qr(m)
            unitaries.append(q.astype(np.complex64))
        return unitaries
    return _generate


# ============================================================================
# KAK Decomposition Tests
# ============================================================================

class TestKAKDecomposition:
    
    def test_kak_identity(self, standard_gates):
        """Identity should give zero KAK coordinates."""
        coords = kak_utils.compute_kak_coords(standard_gates['I'])
        assert jnp.allclose(coords, jnp.zeros(3), atol=1e-5), \
            f"Identity should have zero coords, got {coords}"
    
    def test_kak_cnot(self, standard_gates):
        """CNOT should give (pi/4, 0, 0) coordinates."""
        coords = kak_utils.compute_kak_coords(standard_gates['CNOT'])
        expected = jnp.array([jnp.pi/4, 0, 0])
        assert jnp.allclose(coords, expected, atol=1e-4), \
            f"CNOT should have coords {expected}, got {coords}"
    
    def test_kak_swap(self, standard_gates):
        """SWAP should give (pi/4, pi/4, pi/4) coordinates."""
        coords = kak_utils.compute_kak_coords(standard_gates['SWAP'])
        expected = jnp.array([jnp.pi/4, jnp.pi/4, jnp.pi/4])
        assert jnp.allclose(coords, expected, atol=1e-4), \
            f"SWAP should have coords {expected}, got {coords}"
    
    def test_kak_ordering(self, random_unitaries):
        """KAK coords should satisfy x >= y >= |z|."""
        for u in random_unitaries(5):
            x, y, z = kak_utils.compute_kak_coords(u)
            assert x >= y - 1e-5, f"x >= y violated: {x} < {y}"
            assert y >= abs(z) - 1e-5, f"y >= |z| violated: {y} < {abs(z)}"
    
    def test_kak_weyl_chamber(self, random_unitaries):
        """KAK coords should be in Weyl chamber."""
        for u in random_unitaries(5):
            x, y, z = kak_utils.compute_kak_coords(u)
            assert 0 <= x <= jnp.pi/4 + 1e-5, f"x out of range: {x}"
            assert 0 <= y <= jnp.pi/4 + 1e-5, f"y out of range: {y}"
            assert 0 <= abs(z) <= jnp.pi/4 + 1e-5, f"|z| out of range: {abs(z)}"


# ============================================================================
# Gate Classification Tests
# ============================================================================

class TestGateClassification:
    
    def test_classify_identity(self, standard_gates):
        coords = kak_utils.compute_kak_coords(standard_gates['I'])
        gate_type = kak_utils.classify_gate_type(coords)
        assert gate_type == 'identity', f"Expected 'identity', got '{gate_type}'"
    
    def test_classify_cnot(self, standard_gates):
        coords = kak_utils.compute_kak_coords(standard_gates['CNOT'])
        gate_type = kak_utils.classify_gate_type(coords)
        assert gate_type == 'cnot', f"Expected 'cnot', got '{gate_type}'"
    
    def test_classify_swap(self, standard_gates):
        coords = kak_utils.compute_kak_coords(standard_gates['SWAP'])
        gate_type = kak_utils.classify_gate_type(coords)
        assert gate_type == 'swap', f"Expected 'swap', got '{gate_type}'"
    
    def test_classify_generic(self, random_unitaries):
        u = random_unitaries(1)[0]
        coords = kak_utils.compute_kak_coords(u)
        gate_type = kak_utils.classify_gate_type(coords)
        assert gate_type in ['identity', 'swap', 'cnot', 'generic']


# ============================================================================
# Synthesis Fidelity Tests
# ============================================================================

class TestSynthesisFidelity:
    
    def test_synthesize_identity(self, standard_gates):
        """Identity should synthesize with perfect fidelity."""
        params, loss, gate_count, method = synthesize_gate_smart(
            jnp.array(standard_gates['I']),
            use_analytical=True
        )
        assert loss < 1e-5, f"Identity synthesis failed with loss {loss}"
        assert gate_count == 0, f"Identity should use 0 gates, used {gate_count}"
        assert method == 'analytical', f"Should use analytical method"
    
    def test_synthesize_cnot(self, standard_gates):
        """CNOT should synthesize with high fidelity."""
        params, loss, gate_count, method = synthesize_gate_smart(
            jnp.array(standard_gates['CNOT']),
            use_analytical=True
        )
        assert loss < 1e-3, f"CNOT synthesis failed with loss {loss}"
        assert method == 'analytical', f"Should use analytical for CNOT"
    
    def test_synthesize_random_gates(self, random_unitaries):
        """Random gates should synthesize with good fidelity."""
        for u in random_unitaries(3):
            params, loss, gate_count, method = synthesize_gate_smart(
                jnp.array(u),
                use_analytical=True
            )
            assert loss < 0.01, f"Synthesis failed with loss {loss}"
            assert gate_count <= 3, f"Used too many gates: {gate_count}"
    
    def test_parametric_unitary_consistency(self, standard_gates):
        """Parametric unitary should match Cirq implementation."""
        unitary_fn = create_parametric_unitary_fn()
        
        # Test with zero params (should give identity)
        params = jnp.zeros((4, 2, 3))
        u_jax = unitary_fn(params)
        
        # Build equivalent Cirq circuit
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq_google.SYC(q0, q1),
            cirq_google.SYC(q0, q1),
            cirq_google.SYC(q0, q1)
        ])
        u_cirq = cirq.unitary(circuit)
        
        # Compare (allowing for global phase)
        overlap = abs(np.trace(np.conj(u_cirq).T @ np.array(u_jax))) / 4.0
        assert overlap > 0.999, f"Consistency check failed: overlap = {overlap}"


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    
    @pytest.mark.slow
    def test_batch_kak_computation(self, random_unitaries):
        """Batch KAK should be faster than sequential."""
        import time
        
        unitaries = random_unitaries(100)
        u_stack = jnp.stack([jnp.array(u) for u in unitaries])
        
        # Sequential
        start = time.time()
        for u in unitaries:
            _ = kak_utils.compute_kak_coords(jnp.array(u))
        seq_time = time.time() - start
        
        # Batch
        start = time.time()
        _ = kak_utils.kak_coords_batch(u_stack)
        batch_time = time.time() - start
        
        print(f"\nSequential: {seq_time:.4f}s, Batch: {batch_time:.4f}s")
        assert batch_time < seq_time, "Batch should be faster than sequential"
    
    @pytest.mark.slow
    def test_synthesis_convergence_speed(self, random_unitaries):
        """VQE should converge in reasonable time."""
        import time
        
        u = jnp.array(random_unitaries(1)[0])
        
        start = time.time()
        params, loss, _, _ = synthesize_gate_smart(u, use_analytical=False)
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"Synthesis too slow: {elapsed:.2f}s"
        assert loss < 0.01, f"Poor convergence: loss = {loss}"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    
    def test_cirq_roundtrip(self, standard_gates):
        """Test full synthesis -> Cirq reconstruction -> verification."""
        target_u = standard_gates['CNOT']
        
        # Synthesize
        params, loss, _, _ = synthesize_gate_smart(jnp.array(target_u))
        
        # Reconstruct with Cirq
        q0, q1 = cirq.LineQubit.range(2)
        ops = []
        
        for layer_idx in range(3):  # 3 Sycamore layers
            # Single-qubit gates before SYC
            ops.append(cirq.PhasedXZGate(
                x_exponent=float(params[layer_idx, 0, 0]),
                z_exponent=float(params[layer_idx, 0, 1]),
                axis_phase_exponent=float(params[layer_idx, 0, 2])
            )(q0))
            ops.append(cirq.PhasedXZGate(
                x_exponent=float(params[layer_idx, 1, 0]),
                z_exponent=float(params[layer_idx, 1, 1]),
                axis_phase_exponent=float(params[layer_idx, 1, 2])
            )(q1))
            
            # SYC gate
            ops.append(cirq_google.SYC(q0, q1))
        
        # Final layer (K4)
        ops.append(cirq.PhasedXZGate(
            x_exponent=float(params[3, 0, 0]),
            z_exponent=float(params[3, 0, 1]),
            axis_phase_exponent=float(params[3, 0, 2])
        )(q0))
        ops.append(cirq.PhasedXZGate(
            x_exponent=float(params[3, 1, 0]),
            z_exponent=float(params[3, 1, 1]),
            axis_phase_exponent=float(params[3, 1, 2])
        )(q1))
        
        circuit = cirq.Circuit(ops)
        u_reconstructed = cirq.unitary(circuit)
        
        # Verify fidelity
        overlap = abs(np.trace(np.conj(target_u).T @ u_reconstructed)) / 4.0
        assert overlap > 0.99, f"Roundtrip fidelity too low: {overlap}"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    
    def test_numerical_stability_near_identity(self):
        """Test stability for gates very close to identity."""
        epsilon = 1e-8
        u = np.eye(4, dtype=np.complex64) + epsilon * np.random.randn(4, 4)
        u, _ = np.linalg.qr(u)  # Orthogonalize
        
        coords = kak_utils.compute_kak_coords(jnp.array(u))
        assert jnp.all(jnp.isfinite(coords)), "NaN or Inf in KAK coords"
    
    def test_global_phase_invariance(self, random_unitaries):
        """KAK should be invariant to global phase."""
        u = jnp.array(random_unitaries(1)[0])
        phase = jnp.exp(1j * 0.5)
        
        coords1 = kak_utils.compute_kak_coords(u)
        coords2 = kak_utils.compute_kak_coords(phase * u)
        
        assert jnp.allclose(coords1, coords2, atol=1e-5), \
            "KAK not invariant to global phase"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])