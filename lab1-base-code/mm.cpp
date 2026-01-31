#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include "my_timer.h"

#define NI 4096
#define NJ 4096
#define NK 4096

#define BS 32 // Tile Size
#define UNROLL 4

#define min(a, b) (((a) < (b)) ? (a) : (b))

int i, j, k, ii, jj, kk;

/* Array initialization. */
static void init_array(float C[NI * NJ], float A[NI * NK], float B[NK * NJ])
{

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i * NJ + j] = (float)((i * j + 1) % NI) / NI;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i * NK + j] = (float)(i * (j + 1) % NK) / NK;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i * NJ + j] = (float)(i * (j + 2) % NJ) / NJ;
}

static void print_array_sum(float C[NI * NJ])
{
  int i, j;
  float sum = 0.0;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      sum += C[i * NJ + j];
  printf("sum of C array = %f\n", sum);
}

// =========================================================================
// VARIATION 1: 2D Tiling (i and j loops only)
// =========================================================================
static void kernel_gemm_2d_ij(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{

  // Outer tiled loops (i, j)
  for (i = 0; i < NI; i += BS)
  {
    for (j = 0; j < NJ; j += BS)
    {

      // 1. Beta Scaling (done inside the tile)
      for (ii = i; ii < i + BS; ii++)
      {
        for (jj = j; jj < j + BS; jj++)
        {
          C[ii * NJ + jj] *= beta;
        }
      }

      // 2. Matrix Multiplication
      for (k = 0; k < NK; k++)
      {
        for (ii = i; ii < i + BS; ii++)
        {
          for (jj = j; jj < j + BS;  jj++)
          {
            C[ii * NJ + jj] += alpha * A[ii * NK + k] * B[k * NJ + jj];
          }
        }
      }
    }
  }
}

// =========================================================================
// VARIATION 2: 2D Tiling (i and k loops only)
// =========================================================================
static void kernel_gemm_2d_ik(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{

  // Outer tiled loop (i)
  for (i = 0; i < NI; i += BS)
  {

    // 'j' is NOT tiled, runs fully.
    for (j = 0; j < NJ; j++)
    {

      // 1. Beta Scaling
      for (ii = i; ii < i + BS; ii++)
      {
        C[ii * NJ + j] *= beta;
      }

      // 2. Matrix Multiplication
      // Outer tiled loop (k)
      for (k = 0; k < NK; k += BS)
      {

        // Inner loops
        for (ii = i; ii < i + BS; ii++)
        {
          for (kk = k; kk < k + BS; kk++)
          {
            C[ii * NJ + j] += alpha * A[ii * NK + kk] * B[kk * NJ + j];
          }
        }
      }
    }
  }
}

// =========================================================================
// VARIATION 3: 2D Tiling (j and k loops only)
// =========================================================================
static void kernel_gemm_2d_jk(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{

  // 'i' is NOT tiled, runs fully.
  for (i = 0; i < NI; i++)
  {

    // Outer tiled loop (j)
    for (j = 0; j < NJ; j += BS)
    {

      // 1. Beta Scaling
      for (jj = j; jj < j + BS; jj++)
      {
        C[i * NJ + jj] *= beta;
      }

      // 2. Matrix Multiplication
      // Outer tiled loop (k)
      for (k = 0; k < NK; k += BS)
      {

        // Inner loops
        for (jj = j; jj < j + BS; jj++)
        {
          for (kk = k; kk < k + BS; kk++)
          {
            C[i * NJ + jj] += alpha * A[i * NK + kk] * B[kk * NJ + jj];
          }
        }
      }
    }
  }
}

// =========================================================================
// VARIATION 4: 3D Tiling (i, j, and k loops tiled)
// =========================================================================
static void kernel_gemm_3D(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{

  // Tiling all three (i, j, k) loops
  for (i = 0; i < NI; i += BS)
  {
    for (j = 0; j < NJ; j += BS)
    {

      // 1. Beta Scaling
      for (ii = i; ii < i + BS; ii++)
      {
        for (jj = j; jj < j + BS; jj++)
        {
          C[ii * NJ + jj] *= beta;
        }
      }

      // 2. Matrix Multiplication
      for (k = 0; k < NK; k += BS)
      {

        // Inner Loops: ii -> jj -> kk
        for (ii = i; ii < i + BS; ii++)
        {
          for (jj = j; jj < j + BS; jj++)
          {
            for (kk = k; kk < k + BS; kk++)
            {
              C[ii * NJ + jj] += alpha * A[ii * NK + kk] * B[kk * NJ + jj];
            }
          }
        }
      }
    }
  }
}

// =========================================================================
// VARIATION 5: 3D Tiling Optimized (Moved loops to maximize performance)
// =========================================================================
static void kernel_gemm3D_Optmized(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{

  // Same as variation 4 but with only with optimized loop
  for (i = 0; i < NI; i++)
  {
    for (j = 0; j < NJ; j++)
    {
      C[i * NJ + j] *= beta;
    }
  }

  for (ii = 0; ii < NI; ii += BS)
  {
    for (jj = 0; jj < NJ; jj += BS)
    {
      for (kk = 0; kk < NK; kk += BS)
      {

        for (i = ii; i < ii + BS; i++)
        {

          // Optimization: Loop Order i-k-j (K is middle)
          for (k = kk; k < kk + BS; k++)
          {
            float val_A = alpha * A[i * NK + k];

            for (j = jj; j < jj + BS; j++)
            {
              C[i * NJ + j] += val_A * B[k * NJ + j];
            }
          }
        }
      }
    }
  }
}

// =========================================================================
// VARIATION 6: 3D Tiling Optimized with Vectorization
// =========================================================================
static void kernel_gemm_vect(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{

  for (ii = 0; ii < NI; ii += BS)
  {
    for (jj = 0; jj < NJ; jj += BS)
    {
      // Optimization 1: Initialize/Scale C only once per tile block
      for (i = ii; i < ii + BS; i++)
      {
        for (j = jj; j < jj + BS; j++)
        {
          C[i * NJ + j] *= beta;
        }
      }

      for (kk = 0; kk < NK; kk += BS)
      {
        for (i = ii; i < ii + BS; i++)
        {
          // Cache the row offset of A and C
          int i_NK = i * NK;
          int i_NJ = i * NJ;

          for (k = kk; k < kk + BS; k++)
          {
            // Optimization 2: Pre-calculate alpha * A[i][k]
            float val_A = alpha * A[i_NK + k];
            int k_NJ = k * NJ;

            // Optimization 3: Explicit SIMD with alignment hint
            #pragma omp simd
            for (j = jj; j < jj + BS; j++)
            {
              C[i_NJ + j] += val_A * B[k_NJ + j];
            }
          }
        }
      }
    }
  }
}

// =========================================================================
// Variation 7: 3D Tiling + Optimized i-k-j Order + OpenMP + Vectorization + Unrolling Strat
// =========================================================================
static void kernel_gemm(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{
  int i, j, k;
  int ii, jj, kk;

  // 1. Parallelize the Outer Loops (Tiles)
  // We use collapse(2) to give OpenMP more chunks to distribute to threads.
  #pragma omp parallel for collapse(2) schedule(dynamic) private(ii, jj, kk, i, j, k)
  for (ii = 0; ii < NI; ii += BS)
  {
    for (jj = 0; jj < NJ; jj += BS)
    {

      // Determine the boundaries of the current tile
      int i_max = min(ii + BS, NI);
      int j_max = min(jj + BS, NJ);

      // 2. Beta Scaling (Merged into the tile)
      // Do this first while the cache is fresh.
      if (beta != 1.0f)
      {
        for (i = ii; i < i_max; i++)
        {
          if (beta == 0.0f)
          {
            #pragma omp simd
            for (j = jj; j < j_max; j++)
              C[i * NJ + j] = 0.0f;
          }
          else
          {
            #pragma omp simd
            for (j = jj; j < j_max; j++)
              C[i * NJ + j] *= beta;
          }
        }
      }

      // 3. Tiled Matrix Multiplication
      for (kk = 0; kk < NK; kk += BS)
      {
        int k_max = min(kk + BS, NK);

        // --- REGISTER BLOCKING (4 rows at a time) ---
        // We iterate 'i' by 4.
        for (i = ii; i < i_max - (UNROLL - 1); i += UNROLL)
        {

          for (k = kk; k < k_max; k++)
          {

            // Load 4 values from A (one for each row being processed)
            // We multiply by alpha here to reduce work in the innermost loop.
            float val_a0 = alpha * A[(i + 0) * NK + k];
            float val_a1 = alpha * A[(i + 1) * NK + k];
            float val_a2 = alpha * A[(i + 2) * NK + k];
            float val_a3 = alpha * A[(i + 3) * NK + k];

            #pragma omp simd
            for (j = jj; j < j_max; j++)
            {
              int b_idx = k * NJ + j;
              float val_b = B[b_idx];

              C[(i + 0) * NJ + j] += val_a0 * val_b;
              C[(i + 1) * NJ + j] += val_a1 * val_b;
              C[(i + 2) * NJ + j] += val_a2 * val_b;
              C[(i + 3) * NJ + j] += val_a3 * val_b;
            }
          }
        }

        // 4. Cleanup Loop (for remaining rows if Block Size isn't div by 4)
        for (; i < i_max; i++)
        {
          for (k = kk; k < k_max; k++)
          {
            float val_a = alpha * A[i * NK + k];
            #pragma omp simd
            for (j = jj; j < j_max; j++)
            {
              C[i * NJ + j] += val_a * B[k * NJ + j];
            }
          }
        }
      }
    }
  }
}

int main(int argc, char **argv)
{
  /* Variable declaration/allocation. */
  float *A = (float *)malloc(NI * NK * sizeof(float));
  float *B = (float *)malloc(NK * NJ * sizeof(float));
  float *C = (float *)malloc(NI * NJ * sizeof(float));

  /* Initialize array(s). */
  init_array(C, A, B);

  /* Start timer. */
  timespec timer = tic();

  /* --- RUN THE KERNEL --- */
  /* UNCOMMENT ONE LINE BELOW TO TEST A SPECIFIC STRATEGY */

   kernel_gemm_2d_ij(C, A, B, 1.5, 2.5);        // Strategy 1: Tiling i, j
  // kernel_gemm_2d_ik(C, A, B, 1.5, 2.5);      // Strategy 2: Tiling i, k
  // kernel_gemm_2d_jk(C, A, B, 1.5, 2.5);      // Strategy 3: Tiling j, k
  // kernel_gemm_3D(C, A, B, 1.5, 2.5);         // Strategy 4: Tiling i, j, k
  // kernel_gemm3D_Optmized(C, A, B, 1.5, 2.5);
  // kernel_gemm_vect(C, A, B, 1.5, 2.5);

  //Best Optmized Time
  // kernel_gemm(C, A, B, 1.5, 2.5);

  /* Stop and print timer. */
  toc(&timer, "kernel execution");

  /* Print results. */
  print_array_sum(C);
  printf("Size of Tile: %i\n", BS);
  printf("OpenMP will use up to %d threads\n", omp_get_max_threads());

  /* free memory for A, B, C */
  free(A);
  free(B);
  free(C);

  return 0;
}