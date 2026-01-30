#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "my_timer.h"

#define NI 4096
#define NJ 4096
#define NK 4096

// Note: This CPU contains -> 6 P-Cores (Fast), 8 E-Cores (Slower), Total 20 Threads.
// Using setenv OMP_NUM_THREADS 16 or similar in shell to control threads.

/* Array initialization. */
static void init_array(float C[NI * NJ], float A[NI * NK], float B[NK * NJ])
{
  int i, j;
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

// --- TILING PARAMETERS ---
#define BS 64
#define min(a, b) (((a) < (b)) ? (a) : (b))

// ---------------------------------------------------------------------------
// 1. FINAL OPTIMIZED VERSION (For Competition & Parallel Testing) 3D Tiling + i-k-j Order + OpenMP Parallel + SIMD Vectorization
// ---------------------------------------------------------------------------
static void kernel_gemm(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{
  int i, j, k;
  int ii, jj, kk;

// --- Step 1: Parallel Beta Scaling ---
#pragma omp parallel for private(i, j)
  for (i = 0; i < NI; i++)
  {
    for (j = 0; j < NJ; j++)
    {
      C[i * NJ + j] *= beta;
    }
  }

// --- Step 2: Parallel Tiled Matrix Multiplication ---
#pragma omp parallel for private(jj, kk, i, j, k)
  for (ii = 0; ii < NI; ii += BS)
  {
    for (jj = 0; jj < NJ; jj += BS)
    {
      for (kk = 0; kk < NK; kk += BS)
      {

        // Inner Tile Loops
        for (i = ii; i < min(ii + BS, NI); i++)
        {

          // Optimization: Loop Order i-k-j
          // 'k' loop is middle. This allows us to load A once and reuse it across 'j'.
          for (k = kk; k < min(kk + BS, NK); k++)
          {

            float val_A = alpha * A[i * NK + k];

// 'j' loop is innermost. This is Stride-1 access (Sequential).
// Perfect for Vectorization.
#pragma omp simd
            for (j = jj; j < min(jj + BS, NJ); j++)
            {
              C[i * NJ + j] += val_A * B[k * NJ + j];
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 2. STRATEGY 1: 2D TILING (For Report Comparison)
// ---------------------------------------------------------------------------
static void kernel_gemm_2d(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{
  int i, j, k;
  int ii, jj; // No kk loop here

  // Beta Scaling
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i * NJ + j] *= beta;

  // 2D Tiling (Block i and j)
  for (ii = 0; ii < NI; ii += BS)
  {
    for (jj = 0; jj < NJ; jj += BS)
    {

      // Tile Processing
      for (i = ii; i < min(ii + BS, NI); i++)
      {
        for (j = jj; j < min(jj + BS, NJ); j++)
        {

          float sum = 0.0f;
          for (k = 0; k < NK; k++)
          {
            sum += A[i * NK + k] * B[k * NJ + j];
          }
          C[i * NJ + j] += alpha * sum;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 3. STRATEGY 2: 3D TILING NAIVE (For Report Comparison)
// ---------------------------------------------------------------------------
static void kernel_gemm_3d_naive(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{
  int i, j, k;
  int ii, jj, kk;

  // Beta Scaling
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i * NJ + j] *= beta;

  // 3D Tiling
  for (ii = 0; ii < NI; ii += BS)
  {
    for (jj = 0; jj < NJ; jj += BS)
    {
      for (kk = 0; kk < NK; kk += BS)
      {

        // NAIVE LOOP ORDER: i -> j -> k
        // This is bad because B[k][j] access jumps across memory (Stride-N)
        for (i = ii; i < min(ii + BS, NI); i++)
        {
          for (j = jj; j < min(jj + BS, NJ); j++)
          {

            float sum = 0.0f;
            for (k = kk; k < min(kk + BS, NK); k++)
            {
              sum += A[i * NK + k] * B[k * NJ + j];
            }
            C[i * NJ + j] += alpha * sum;
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

  // UNCOMMENT THE VERSION YOU WANT TO TEST:

  // 1. Final Version (For Competition & Parallel Testing)
  kernel_gemm(C, A, B, 1.5, 2.5);

  // 2. 2D Tiling Strategy (For Report - Strategy 1)
  // kernel_gemm_2d(C, A, B, 1.5, 2.5);

  // 3. 3D Naive Strategy (For Report - Strategy 2)
  // kernel_gemm_3d_naive(C, A, B, 1.5, 2.5);

  /* Stop and print timer. */
  toc(&timer, "kernel execution");

  /* Print results. */
  print_array_sum(C);
  printf("OpenMP will use up to %d threads\n", omp_get_max_threads());

  /* free memory for A, B, C */
  free(A);
  free(B);
  free(C);

  return 0;
}