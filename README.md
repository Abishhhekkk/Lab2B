# 2D Convolution Performance Optimization Using AVX2 SIMD

**Course:** ECE 662  
**Laboratory:** Performance Comparison of 2D Convolution Using SIMD  
**Language:** C  
**Optimization:** Intel AVX2

---

## 1. Project Overview

This project evaluates the performance of 2D convolution using two different implementations:

- **Naive implementation:** Conventional scalar computation using nested loops.
- **SIMD implementation:** Parallel computation using Intel AVX2 intrinsics.

The primary objective is to compare execution time while verifying that both implementations produce identical output.

## 2. Project Structure

```text
2D-Convolution-SIMD/
│
├── conv2d.c
│
├── README.md
│
└── results/
    └── execution_results.png
```

| File | Description |
|---|---|
| `conv2d.c` | Contains the naive and AVX2 implementations, along with correctness verification and benchmarking. |
| `README.md` | Project documentation and execution instructions. |
| `results/execution_results.png` | Screenshot of program output and performance measurements. |

## 3. Experimental Configuration

| Parameter | Configuration |
|---|---|
| Input dimensions | 2048 × 2048 |
| Kernel dimensions | 4 × 4 |
| Output dimensions | 2045 × 2045 |
| Input data type | `int8_t` |
| Output data type | `int32_t` |
| SIMD architecture | AVX2 |
| Compiler | GCC |
| Optimization flag | `-O3` |

## 4. Implementation Modules

### 4.1 Naive Convolution

The baseline implementation performs 2D convolution using nested loops. Each output element is computed through scalar multiplication and accumulation over the kernel window.

### 4.2 AVX2 SIMD Convolution

The optimized implementation uses AVX2 intrinsics to accelerate the convolution operation.

The implementation consists of the following stages:

1. **Data loading:** Loads four 8-bit input and kernel elements.
2. **Data widening:** Converts signed 8-bit elements into signed 16-bit elements.
3. **Parallel multiplication:** Uses `_mm256_mullo_epi16` to multiply corresponding elements.
4. **Accumulation:** Extracts the four valid products and accumulates them into a 32-bit sum.

### 4.3 Correctness Verification

The output of the SIMD implementation is compared against the naive implementation.

The program reports:

```text
Correct: 1
```

This indicates that both implementations produce matching results.

### 4.4 Performance Benchmarking

Both implementations are benchmarked to measure their execution times.

## 5. Build and Execution

### Prerequisites

- GCC compiler
- AVX2-compatible processor
- Windows or Linux environment

### Compilation

```bash
gcc -std=c11 -O3 -mavx2 -o conv2d conv2d.c
```

### Execution

**Windows PowerShell**

```powershell
.\conv2d.exe
```

**Linux**

```bash
./conv2d
```

## 6. Performance Evaluation

Five independent execution measurements were collected.

| Run | Naive (s) | AVX2 (s) |
|---:|---:|---:|
| 1 | 0.067 | 0.053 |
| 2 | 0.067 | 0.052 |
| 3 | 0.070 | 0.053 |
| 4 | 0.079 | 0.056 |
| 5 | 0.069 | 0.054 |
| **Average** | **0.0704** | **0.0536** |

### Performance Summary

| Metric | Result |
|---|---:|
| Average naive runtime | 0.0704 s |
| Average AVX2 runtime | 0.0536 s |
| Speedup | **1.31×** |
| Runtime reduction | **23.86%** |
| Correctness | **Passed (5/5 runs)** |

Speedup is calculated as:

\[
\text{Speedup} =
\frac{\text{Naive Average Runtime}}
{\text{AVX2 Average Runtime}}
\]

\[
= \frac{0.0704}{0.0536}
\approx 1.31\times
\]

## 7. Execution Results

The following screenshot shows the program output, correctness verification, and execution-time measurements.

![Execution Results](results/execution_results.png)

## 8. Conclusion

The experiment demonstrates the performance benefits of SIMD optimization for 2D convolution. The AVX2 implementation achieved an average speedup of approximately **1.31×** over the naive implementation, reducing execution time by approximately **23.86%** while maintaining output correctness.

---

**Implementation:** C with Intel AVX2 intrinsics  
**Course:** ECE 662
