# Tiled GEMM (OpenMP + SIMD) — Notes

This project implements a matrix multiply kernel in C:

**C = alpha * A * B + beta * C**

using:
- **Cache tiling (blocking)** to improve data locality
- **OpenMP parallelization** to use multiple CPU threads
- **SIMD vectorization** on the inner loop (compiler-assisted via `#pragma omp simd`)

---

## Matrices and Layout

- A is **NI x NK**
- B is **NK x NJ**
- C is **NI x NJ**
- All matrices are stored as **1D arrays** in **row-major** order:

Index mapping:
- `A[i][k]` → `A[i * NK + k]`
- `B[k][j]` → `B[k * NJ + j]`
- `C[i][j]` → `C[i * NJ + j]`

---

## Why Tiling (Blocking) Helps

For large matrices (e.g., 4096 x 4096), working sets are much larger than CPU cache.

**Tiling** breaks the computation into **BS x BS** blocks so that:
- a chunk of A and B is reused from cache
- C updates happen in a locality-friendly region
- fewer cache misses → better performance

`BS` is the block size (currently `64`).

---

## Kernel Structure (What the code does)

### 1) Beta scaling (done once)
We first apply:

**C = beta * C**

This is done separately so we **don’t multiply by beta inside the hottest inner loop**.

### 2) Tiled multiply-add
Then we do:

**C += alpha * A * B**

using 3D tiling loops over:
- `ii` for rows of C (and A)
- `jj` for cols of C (and B)
- `kk` for the shared dimension

Inner loop order is **i-k-j**:

- `i` selects a row in A and C
- `k` selects one element in A and a row in B
- `j` walks across the row in B and C (contiguous access)

This is good because:
- `B[k * NJ + j]` is **stride-1** in `j` (sequential)
- `C[i * NJ + j]` is **stride-1** in `j`
- that makes the `j` loop a great SIMD/vectorization target

We also preload:

`val_A = alpha * A[i * NK + k]`

so that A’s scalar value is reused across the full `j` range.

---

## Parallelization (OpenMP)

The code uses OpenMP `parallel for` for:
- Beta scaling loop over `i`
- Outer tiling loop over `ii` (distributes tiles across threads)

You can control thread count using:

- Bash:
  - `export OMP_NUM_THREADS=16`

- Or in one command:
  - `OMP_NUM_THREADS=16 ./a.out`

To see max threads at runtime:
- `omp_get_max_threads()`

---

## SIMD Vectorization

The innermost `j` loop uses:

`#pragma omp simd`

This tells the compiler to vectorize the loop (when safe), usually generating AVX/AVX2/AVX-512 depending on CPU and compiler flags.

---

## Tile Boundary Safety

The macro:

`min(ii + BS, NI)` (and similarly for NJ/NK)

prevents out-of-bounds access when NI/NJ/NK are not exact multiples of BS.

---

## Build Notes (Typical)

Example (GCC/Clang):
- `-O3` for optimization
- `-fopenmp` for OpenMP
- optional: `-march=native` to enable best SIMD for your CPU

Example:
- `gcc -O3 -fopenmp -march=native gemm.c my_timer.c -o gemm`

---

## Functions

- `init_array()`  
  Initializes A, B, C with deterministic values.

- `kernel_gemm()`  
  Main optimized kernel: beta scaling + 3D tiling + OpenMP + SIMD.

- `print_array_sum()`  
  Sums C to provide a cheap correctness signal (instead of printing all values).

---

## System Note (Your CPU)
Your note: CPU has mixed cores (P-cores + E-cores). Performance may change depending on thread count because the scheduler may place threads on slower E-cores.
