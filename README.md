# JAX-Accelerated Quantum Circuit Compilation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-enabled-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance quantum circuit compilation framework for Google Sycamore architecture, achieving **27.1x speedup** over baseline using JAX-accelerated tensor processing and intelligent KAK decomposition.

## 🚀 Key Features

- **27.1x Compilation Speedup**: 9,400 gates/second vs 350 baseline
- **High Fidelity**: F > 0.999 with KAK-guided initialization
- **Parallel Architecture**: Map-Reduce + ProcessPoolExecutor for GIL-free execution
- **Smart Synthesis**: Analytical shortcuts for known gates, VQE for arbitrary unitaries
- **Production-Ready**: Comprehensive test suite, error handling, and benchmarking

## 📊 Performance Metrics

| Metric | Sequential | Parallel JAX | Speedup |
|--------|-----------|--------------|---------|
| Time (72K gates) | 206s | 7.62s | **27.1x** |
| Throughput | 350 gates/s | 9,400 gates/s | **26.9x** |
| Fidelity | 0.9999 | 0.9999 | Equal |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Input Circuit                          │
│              (Arbitrary 2-qubit gates)                   │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼──────────┐
         │   Gate Classification │
         │    (KAK Analysis)     │
         └───────────┬───────────┘
                     │
        ┌────────────┴─────────────┐
        │                          │
   ┌────▼─────┐           ┌───────▼────────┐
   │ Analytical│           │  VQE Pipeline  │
   │ (CNOT,    │           │  (Arbitrary)   │
   │  SWAP,    │           │                │
   │  Identity)│           │  Multi-start   │
   └────┬──────┘           │  + Early Stop  │
        │                  └────────┬────────┘
        │                           │
        └────────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │   Parallel Map-Reduce    │
         │  (ProcessPoolExecutor)   │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │  Sycamore Circuit Output │
         │    (PhasedXZ + SYC)      │
         └──────────────────────────┘
```

## 🔧 Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, for maximum performance)

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/quantum-compilation.git
cd quantum-compilation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest test_suite.py -v

# Run benchmark
python compilation_sycamore.py
```

### Docker Setup (Recommended)

```bash
docker build -t quantum-compiler .
docker run --gpus all quantum-compiler
```

## 📖 Usage

### Basic Example

```python
import cirq
import cirq_google
from compilation_sycamore import parallel_sycamore_compilation

# Create a quantum circuit
qubits = cirq.LineQubit.range(4)
circuit = cirq.Circuit([
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.CNOT(qubits[2], qubits[3])
])

# Compile to Sycamore gateset
compiled_circuit = parallel_sycamore_compilation(circuit)

print(f"Original gates: {len(list(circuit.all_operations()))}")
print(f"Compiled gates: {len(list(compiled_circuit.all_operations()))}")
```

### Advanced: Custom VQE Parameters

```python
from optimized_vqe import synthesize_gate_smart
import numpy as np

# Define target unitary (e.g., random 2-qubit gate)
m = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
target_u, _ = np.linalg.qr(m)

# Synthesize with custom settings
params, loss, gate_count, method = synthesize_gate_smart(
    target_u,
    use_analytical=True  # Use shortcuts for known gates
)

print(f"Method: {method}")
print(f"Gate count: {gate_count}")
print(f"Fidelity: {1 - loss:.6f}")
```

## 🧪 Testing

```bash
# Run all tests
pytest test_suite.py -v

# Run specific test categories
pytest test_suite.py -v -k "TestKAK"  # KAK decomposition tests
pytest test_suite.py -v -k "TestSynthesis"  # Synthesis tests

# Run performance benchmarks
pytest test_suite.py -v -m slow

# Generate coverage report
pytest --cov=. --cov-report=html
```

## 📈 Benchmarking

```bash
# Standard benchmark (72K gates)
python compilation_sycamore.py

# Custom benchmark
python -c "
from compilation_sycamore import benchmark_compilation
benchmark_compilation(
    n_qubits=10,
    depth=100,
    gate_type='random'
)
"
```

## 🔬 Technical Details

### KAK Decomposition

The framework uses Cartan (KAK) decomposition to extract topological invariants from arbitrary two-qubit unitaries:

```
U = (k1_a ⊗ k1_b) · exp(i(x·XX + y·YY + z·ZZ)) · (k2_a ⊗ k2_b)
```

Where `(x, y, z)` are interaction coefficients in the Weyl chamber: `0 ≤ z ≤ y ≤ x ≤ π/4`

### Variational Optimization

For generic gates, we use a parameterized ansatz:

```
U(θ) = K4 · SYC · K3 · SYC · K2 · SYC · K1
```

Where each `Ki` is a product of single-qubit `PhasedXZ` gates. Optimization uses Adam with:
- **Smart initialization** from KAK coordinates
- **Multi-start strategy** (1 KAK + N random)
- **Early stopping** (patience=20, tol=1e-6)

### Parallelization Strategy

1. **Task Decomposition**: Each gate is an independent task
2. **Dynamic Load Balancing**: ProcessPoolExecutor manages worker pool
3. **Batched Computation**: JAX vmaps for SIMD vectorization
4. **Minimal IPC**: Only pass NumPy arrays between processes

## 📚 API Reference

### Core Functions

#### `parallel_sycamore_compilation(circuit, executor=None)`
Compiles a Cirq circuit to Sycamore gateset using parallel processing.

**Parameters:**
- `circuit` (cirq.Circuit): Input circuit
- `executor` (ProcessPoolExecutor, optional): Reusable executor

**Returns:**
- cirq.Circuit: Compiled circuit with only SYC and PhasedXZ gates

#### `synthesize_gate_smart(target_u, use_analytical=True)`
Synthesizes a single 2-qubit unitary.

**Parameters:**
- `target_u` (array): 4×4 unitary matrix
- `use_analytical` (bool): Use shortcuts for known gates

**Returns:**
- `params` (array): Gate parameters (4,2,3)
- `loss` (float): Final infidelity
- `gate_count` (int): Number of Sycamore gates used
- `method` (str): 'analytical' or 'vqe'

### Utility Functions

#### `kak_utils.compute_kak_coords(unitary)`
Extracts KAK interaction coefficients.

#### `kak_utils.classify_gate_type(kak_coords)`
Classifies gate as 'identity', 'cnot', 'swap', or 'generic'.

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linters
black .
flake8 .
mypy .
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@article{khadivi2024jax,
  title={High-Performance Quantum Compilation via JAX-Accelerated Tensor Pipelining},
  author={Khadivi, Ata},
  journal={arXiv preprint},
  year={2024}
}
```

## 🔗 Links

- **Documentation**: [https://docs.example.com](https://docs.example.com)
- **Paper**: [arXiv:XXXX.XXXXX](https://arxiv.org)
- **Author**: [Ata Khadivi](https://linkedin.com/in/atakhadivi)
- **GitHub**: [github.com/atakhadiviom/Quantum-AI-Career](https://github.com/atakhadiviom/Quantum-AI-Career)

## 🙏 Acknowledgments

- Google Cirq team for the excellent quantum framework
- JAX team for the high-performance computing tools
- Quantum computing community for inspiration

---

**Status**: Active Development | **Stability**: Beta | **Version**: 0.2.0