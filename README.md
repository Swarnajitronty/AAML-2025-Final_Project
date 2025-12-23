# AAML 2025 Final Project: Pruned Wav2Letter Hardware Acceleration

## Student Information
- **Name:** Swarnajit Bhattacharya
- **Student ID:** 314540038

## Project Overview
This repository contains the implementation of a Custom Function Unit (CFU) designed to accelerate the pruned Wav2Letter model for Automatic Speech Recognition (ASR) tasks. The project focuses on minimizing inference latency while maintaining model accuracy above 72%.

## Project Website
For complete project specifications and requirements, visit:  
[AAML 2025 Final Project - Pruned Wav2Letter](https://nycu-caslab.github.io/AAML2025/project/final_project.html)

## Repository Structure

### Core Files

#### `cfu.v`
The main Verilog module implementing the Custom Function Unit (CFU) for hardware acceleration. This file contains:
- Custom hardware accelerator logic
- Interface with the RISC-V processor
- Optimized operations for convolution and other compute-intensive layers
- SIMD and systolic array implementations (if applicable)

#### `eval.py`
Python evaluation script for testing the accelerated model performance. Features:
- Automated testing of the Wav2Letter model on the provided dataset
- Latency measurement and reporting
- Accuracy calculation using Word Error Rate (WER)
- Serial communication with the FPGA board via UART

**Usage:**
```bash
python eval.py --port /dev/ttyUSB1
```

#### `Makefile`
Build automation file for compiling and deploying the project to the FPGA. Key targets:
- `make prog`: Programs the FPGA with the design
- `make load`: Loads the software onto the board
- Compilation of C/C++ sources with TensorFlow Lite integration
- Verilog synthesis and bitstream generation

#### `y_labels.csv`
Ground truth labels for the evaluation dataset. Contains:
- Reference transcriptions for audio samples
- Used by `eval.py` to calculate Word Error Rate (WER)
- Benchmark data for accuracy validation

### Directories

#### `perf_samples/`
Performance testing audio samples directory. Contains:
- Audio files for inference testing
- Speech samples used for latency measurement
- Evaluation dataset for accuracy assessment

#### `src/`
Source code directory containing:
- **TensorFlow Lite kernel implementations:**
  - `tensorflow/lite/kernels/internal/reference/integer_ops/conv.h` - Optimized convolution operations
  - `tensorflow/lite/kernels/internal/reference/leaky_relu.h` - LeakyReLU activation function
- **Wav2Letter model files:**
  - `wav2letter/model/` - Model definition and converted header files
- **Application code:**
  - Main inference loop
  - CFU integration code
  - Menu system and golden test implementations

## Key Features
- Hardware-accelerated inference for Wav2Letter ASR model
- Pruned (50% sparsity) and INT8 quantized model
- Custom FPGA-based acceleration using CFU
- Optimized convolution and activation operations
- Real-time speech recognition capability

## Performance Metrics
The project is evaluated based on:
- **Accuracy:** Must maintain ≥72% Word Error Rate performance
- **Latency:** Minimize inference time per audio sample
- **Ranking:** Performance compared against other implementations

## Build and Run Instructions

### Prerequisites
```bash
pip install numpy pyserial tqdm jiwer
```

### Building the Project
```bash
cd ${CFU_ROOT}/proj/AAML-2025-Project
make prog && make load
```

### Running Golden Test
1. After loading, press `3` → `w` → `g` in the LiteX terminal
2. Verify output matches expected results

### Running Performance Evaluation
1. Close the LiteX terminal to free the UART port
2. Execute the evaluation script:
```bash
python eval.py --port /dev/ttyUSB1
```

## Model Information
- **Architecture:** Wav2Letter (Pruned)
- **Quantization:** INT8
- **Sparsity:** 50%
- **Source:** ARM ML-Zoo
- **Task:** Automatic Speech Recognition (ASR)

## References
- [AAML 2025 Course Website](https://nycu-caslab.github.io/AAML2025/)
- [ARM ML-Zoo Repository](https://github.com/ARM-software/ML-Zoo)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)

## License
This project is part of the AAML 2025 course.

---
© 2025 Swarnajit Bhattacharya
