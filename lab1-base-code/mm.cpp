#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "my_timer.h"

#define NI 4096
#define NJ 4096
#define NK 4096

#define BS 128 //Tile Size

#define min(a, b) (((a) < (b)) ? (a) : (b))



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

// =========================================================================
// VARIATION 1: 2D Tiling (i and j loops only)
// =========================================================================
static void kernel_gemm_2d_ij(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{
  int i, j, k, ii, jj;

  // Outer tiled loops (i, j)
  for (i = 0; i < NI; i += BS)
  {
    for (j = 0; j < NJ; j += BS)
    {

      // 1. Beta Scaling (done inside the tile)
      for (ii = i; ii < min(i + BS, NI); ii++)
      {
        for (jj = j; jj < min(j + BS, NJ); jj++)
        {
          C[ii * NJ + jj] *= beta;
        }
      }

      // 2. Matrix Multiplication
      // Note: 'k' loop runs fully (0 to NK). This causes cache thrashing.
      for (k = 0; k < NK; k++)
      {
        for (ii = i; ii < min(i + BS, NI); ii++)
        {
          for (jj = j; jj < min(j + BS, NJ); jj++)
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
  int i, j, k, ii, kk;

  // Outer tiled loop (i)
  for (i = 0; i < NI; i += BS)
  {

    // 'j' is NOT tiled, runs fully.
    for (j = 0; j < NJ; j++)
    {

      // 1. Beta Scaling
      for (ii = i; ii < min(i + BS, NI); ii++)
      {
        C[ii * NJ + j] *= beta;
      }

      // 2. Matrix Multiplication
      // Outer tiled loop (k)
      for (k = 0; k < NK; k += BS)
      {

        // Inner loops
        for (ii = i; ii < min(i + BS, NI); ii++)
        {
          for (kk = k; kk < min(k + BS, NK); kk++)
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
  int i, j, k, jj, kk;

  // 'i' is NOT tiled, runs fully.
  for (i = 0; i < NI; i++)
  {

    // Outer tiled loop (j)
    for (j = 0; j < NJ; j += BS)
    {

      // 1. Beta Scaling
      for (jj = j; jj < min(j + BS, NJ); jj++)
      {
        C[i * NJ + jj] *= beta;
      }

      // 2. Matrix Multiplication
      // Outer tiled loop (k)
      for (k = 0; k < NK; k += BS)
      {

        // Inner loops
        for (jj = j; jj < min(j + BS, NJ); jj++)
        {
          for (kk = k; kk < min(k + BS, NK); kk++)
          {
            C[i * NJ + jj] += alpha * A[i * NK + kk] * B[kk * NJ + jj];
          }
        }
      }
    }
  }
}

// =========================================================================
// VARIATION 4: 3D Tiling (Naive Loop Order)
// =========================================================================
static void kernel_gemm_3D(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;

  // Tiling all three (i, j, k) loops
  for (i = 0; i < NI; i += BS)
  {
    for (j = 0; j < NJ; j += BS)
    {

      // 1. Beta Scaling
      for (ii = i; ii < min(i + BS, NI); ii++)
      {
        for (jj = j; jj < min(j + BS, NJ); jj++)
        {
          C[ii * NJ + jj] *= beta;
        }
      }

      // 2. Matrix Multiplication
      for (k = 0; k < NK; k += BS)
      {

        // Inner Loops: ii -> jj -> kk
        // This is "Naive" because accessing B[kk][jj] inside the innermost loop
        // might not be contiguous if loop orders aren't perfect.
        for (ii = i; ii < min(i + BS, NI); ii++)
        {
          for (jj = j; jj < min(j + BS, NJ); jj++)
          {
            for (kk = k; kk < min(k + BS, NK); kk++)
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
// VARIATION 5: 3D Tiling Optimized (Naive Loop Order)
// =========================================================================
static void kernel_gemm3D_Optmized(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;

  // Parallel Beta Scaling

  for (i = 0; i < NI; i++)
  {
    for (j = 0; j < NJ; j++)
    {
      C[i * NJ + j] *= beta;
    }
  }

  // Parallel Tiled Multiplication

  for (ii = 0; ii < NI; ii += BS)
  {
    for (jj = 0; jj < NJ; jj += BS)
    {
      for (kk = 0; kk < NK; kk += BS)
      {

        for (i = ii; i < min(ii + BS, NI); i++)
        {

          // Optimization: Loop Order i-k-j (K is middle)
          for (k = kk; k < min(kk + BS, NK); k++)
          {
            float val_A = alpha * A[i * NK + k];

            // J is innermost (Stride-1 access) + SIMD

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

static void kernel_gemm_vect(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{
  int ii, jj, kk, i, j, k;

  // Use a single parallel region to reduce fork/join overhead

  for (ii = 0; ii < NI; ii += BS)
  {
    for (jj = 0; jj < NJ; jj += BS)
    {
      // Optimization 1: Initialize/Scale C only once per tile block
      for (i = ii; i < min(ii + BS, NI); i++)
      {
        for (j = jj; j < min(jj + BS, NJ); j++)
        {
          C[i * NJ + j] *= beta;
        }
      }

      for (kk = 0; kk < NK; kk += BS)
      {
        for (i = ii; i < min(ii + BS, NI); i++)
        {
          // Cache the row offset of A and C
          int i_NK = i * NK;
          int i_NJ = i * NJ;

          for (k = kk; k < min(kk + BS, NK); k++)
          {
            // Optimization 2: Pre-calculate alpha * A[i][k]
            float val_A = alpha * A[i_NK + k];
            int k_NJ = k * NJ;

// Optimization 3: Explicit SIMD with alignment hint
#pragma omp simd
            for (j = jj; j < min(jj + BS, NJ); j++)
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
// 3D Tiling + Optimized i-k-j Order + OpenMP + Vectorization
// =========================================================================
static void kernel_gemm_final(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;

// Parallel Beta Scaling
#pragma omp parallel for private(i,j)
  for (i = 0; i < NI; i++)
  {
    for (j = 0; j < NJ; j++)
    {
      C[i * NJ + j] *= beta;
    }
  }

// Parallel Tiled Multiplication
#pragma omp parallel for private(ii,jj, kk, k)
  for (ii = 0; ii < NI; ii += BS)
  {
    for (jj = 0; jj < NJ; jj += BS)
    {
      for (kk = 0; kk < NK; kk += BS)
      {

        for (i = ii; i < min(ii + BS, NI); i++)
        {

          // Optimization: Loop Order i-k-j (K is middle)
          for (k = kk; k < min(kk + BS, NK); k++)
          {
            float val_A = alpha * A[i * NK + k];

// J is innermost (Stride-1 access) + SIMD
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

  //kernel_gemm_2d_ij(C, A, B, 1.5, 2.5);        // Strategy 1: Tiling i, j
  //kernel_gemm_2d_ik(C, A, B, 1.5, 2.5);      // Strategy 2: Tiling i, k
  //kernel_gemm_2d_jk(C, A, B, 1.5, 2.5);      // Strategy 3: Tiling j, k
   //kernel_gemm_3D(C, A, B, 1.5, 2.5);         // Strategy 4: Tiling i, j, k (Naive)
  //kernel_gemm3D_Optmized(C, A, B, 1.5, 2.5);
  // kernel_gemm_vect(C, A, B, 1.5, 2.5);

  // FINAL SUBMISSION (Fastest)
   kernel_gemm_final(C, A, B, 1.5, 2.5);

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