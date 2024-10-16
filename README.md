# Kernel BCPNN Analysis

This repository contains the analysis and optimization of key CUDA kernels in the Bayesian Confidence Propagation Neural Network (BCPNN) model. The focus is on improving the performance of five crucial kernels within the population and projection layers. Both original and optimized versions are provided for comparison and verification.

## Overview

The BCPNN model is inspired by Bayesian inference and Hebbian learning, used to study associative memory and unsupervised learning. It mimics the mammalian cortex structure using hypercolumns and minicolumns.

## Experiments

Experiments were conducted using:
- **Hardware:** NVIDIA RTX 2070 GPU, Intel Core i5-9600K CPU, 32GB RAM
- **Software:** CUDA 12.6, C++17 standard, NVIDIA Nsight Compute 2024.3.1.0

Performance profiling was done using Nsight Compute to evaluate execution characteristics.

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA toolkit installed

### Setup

1. Adjust the architecture in the Makefile (`-arch=sm_75`) based on your GPU.
2. Compile the code:
   ```bash
   make

### Running
Execute the main program:
```bash
./bin/main
```

### Profiling
To profile kernel performance:
```bash
ncu --set full -o outputfile -f ./bin/main
```

### Verification
The code verifies that optimizations do not compromise computational correctness by comparing the outputs of original and optimized kernels.


### File Structure

- `src/` : Source code for the BCPNN CUDA kernels
- `bin/` : Compiled executables
- `BCPNN_full/` : Full implementation of the BCPNN model
- `BCPNN_full_optimized/` : Full implementation of the BCPNN model optimized
- `Makefile` : Makefile for building the project
- `README.md` : Project documentation
- `*.ncu-rep` : NVIDIA Nsight Compute profiling reports

### Run full BCPNN model
to run the full BCPNN model, you need to have GPU with architecture `sm_75`. Then, run it with
```bash
./mnistmain mnistmain.par
```


