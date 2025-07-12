# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Initial development setup (installs dependencies and builds Triton)
make dev-install

# Install requirements only
make dev-install-requires

# Install Triton in development mode
make dev-install-triton

# Build with custom LLVM
make dev-install-llvm
```

### Building
```bash
# Incremental build (uses ninja)
make all

# Build triton-opt tool specifically
make triton-opt
```

### Testing
```bash
# Run all tests (requires GPU)
make test

# Run tests without GPU
make test-nogpu

# Individual test suites
make test-lit          # MLIR lit tests
make test-cpp          # C++ unit tests  
make test-unit         # Python unit tests
make test-regression   # Regression tests
make test-interpret    # Interpreter tests
make test-gluon        # Gluon dialect tests
make test-proton       # Proton profiler tests

# Run specific Python test with custom parallelism
cd python/test/unit && python -m pytest -s -n 8 language/test_core.py
```

### Code Quality
```bash
# Format Python code (yapf/autopep8 config in pyproject.toml)
yapf --in-place --recursive python/

# Lint with ruff (config in pyproject.toml)
ruff check python/

# Type checking with mypy (config in pyproject.toml)
mypy python/triton/knobs.py
```

## Architecture Overview

Triton is an MLIR-based compiler for GPU kernels with a multi-dialect architecture:

### Core Compilation Pipeline
```
Python → Triton IR → TritonGPU IR → [Backend IR] → LLVM IR → GPU Assembly
```

### Key Dialects (lib/Dialect/, include/triton/Dialect/)
- **Triton (`tt`)**: High-level tensor operations
- **TritonGPU (`ttg`)**: GPU-aware operations with memory layouts  
- **TritonNvidiaGPU (`ttng`)**: NVIDIA-specific optimizations (TMA, WGMMA)
- **Gluon**: Experimental high-performance programming model
- **TritonInstrument**: Debugging and profiling support

### Backend Architecture (third_party/)
- **NVIDIA** (`third_party/nvidia/`): CUDA/PTX generation, Tensor Core support
- **AMD** (`third_party/amd/`): HIP/ROCm generation, WMMA/MFMA support
- **Pluggable system**: Backends implement common interface in `python/triton/backends/`

### Python API Structure (python/triton/)
- **language/**: Triton DSL - core operations (`core.py`), math functions (`math.py`)
- **runtime/**: JIT compilation (`jit.py`), autotuning (`autotuner.py`), drivers
- **compiler/**: Compilation orchestration and error handling
- **backends/**: Backend implementations and target configuration

### Key Transforms (lib/Dialect/TritonGPU/Transforms/)
- **AccelerateMatmul**: Matrix multiplication optimization using tensor cores
- **Pipeline**: Software pipelining for latency hiding
- **Coalesce**: Memory access optimization
- **WarpSpecialization**: Advanced execution patterns

## Development Patterns

### Adding New Operations
1. Define operation in appropriate `.td` file (include/triton/Dialect/*/IR/)
2. Implement lowering in conversion passes (lib/Conversion/)
3. Add optimization patterns in transforms (lib/Dialect/*/Transforms/)
4. Add Python frontend support (python/triton/language/)

### Backend Development
- Implement `BaseBackend` interface in `python/triton/backends/compiler.py`
- Add dialect support in `third_party/your_backend/`
- Integrate with main CMake build system

### Testing Patterns
- **MLIR tests**: Use `// CHECK:` patterns in `.mlir` files in `test/`
- **Python tests**: Use pytest framework in `python/test/`
- **Update golden samples**: `make golden-samples` regenerates test expectations

## Debugging

### Environment Variables (see python/triton/knobs.py)
- `MLIR_ENABLE_DUMP=1`: Dump IR before every MLIR pass
- `LLVM_IR_ENABLE_DUMP=1`: Dump LLVM IR before passes  
- `TRITON_INTERPRET=1`: Use interpreter instead of GPU
- `TRITON_REPRODUCER_PATH=<path>`: Generate MLIR reproducer files
- `TRITON_KERNEL_DUMP=1`: Dump compilation stages and final assembly
- `TRITON_ALWAYS_COMPILE=1`: Force recompilation (ignore cache)
- `MLIR_ENABLE_TIMING=1`: Show MLIR pass timing
- `LLVM_ENABLE_TIMING=1`: Show LLVM pass timing

### Kernel Override Workflow
```bash
export TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./dump
# Run once to dump stages, then:
export TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=./override
# Copy and modify stages, run again to test changes
```

## File Organization

### Core Implementation
- `include/triton/`: C++ headers for dialects, passes, utilities
- `lib/`: C++ implementation of dialects, conversions, transforms, analysis
- `python/src/`: C++ Python bindings (Pybind11)

### Testing and Tools  
- `test/`: MLIR lit tests organized by dialect
- `python/test/`: Python pytest test suites
- `bin/`: Command-line tools (triton-opt, etc.)
- `utils/`: Development utilities (test generation, etc.)

### Documentation and Examples
- `docs/`: Sphinx documentation source
- `python/tutorials/`: Example kernels and tutorials
- `python/triton_kernels/`: Reference kernel implementations

This codebase uses a sophisticated multi-level IR approach with hardware-specific backends, extensive optimization passes, and comprehensive testing infrastructure for GPU kernel compilation.