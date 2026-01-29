#include <stdio.h>
#include <stdlib.h>
#include <omp.h>



#include "my_timer.h"

#define NI 4096
#define NJ 4096
#define NK 4096

//* Note: This CPU contains -> 6 P-Cores (Fast) , 8 E-Cores (Slower),  6 P-Cores + 8 E-Cores = 20 Threads in Total

//! setenv OMP_NUM_THREADS 16 Setting usage of threads 

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

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static

    void
    print_array(float C[NI * NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      printf("C[%d][%d] = %f\n", i, j, C[i * NJ + j]);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array_sum(float C[NI * NJ])
{
  int i, j;

  float sum = 0.0;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      sum += C[i * NJ + j];

  printf("sum of C array = %f\n", sum);
}


//* Tiling (Do mulitplication in small chunks that fit in the cache), therefore the CPU reuses data instead of redoing the data again

#define BS 64 //*Defining new block size, 2048 x 2048 =  ~16 MB, They do not fit in the CPU Cache, so break down matrices down

#define min(a, b) (((a) < (b)) ? (a) : (b)) //* Logic:  If a < b return a, if false return b (Helper macro to ensure we don't go out of bounds)

int i, j, k;    // Inner loop counters
int ii, jj, kk; // Outer loop counters
float A_val;

//! kernel_gemm function with 3D Tiling, OpenMP, and Vectorization
static void kernel_gemm(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta) //* => (C = alpha*A*B + beta*C)
{

  //* Frist scaling Matrix C by beta and using parallezation to quickly compute the first part of the computation and it will not be included in the main for loop
  //* Privatizing 'j' prevents a Race Condition so threads do not overwrite the shared variable.(ex: Thread A writes to j and Thread B will overwrite j before thread A is done)

  //* 1. Handle Beta Scaling (Matrix C = C * beta)
  //* We do this first so we don't have to multiply by beta inside the critical path

  //*Paralization
  #pragma omp parallel for private(i, j)
  for (i = 0; i < NI; i++)
  {
    for (j = 0; j < NJ; j++)
    {

      C[i * NJ + j] *= beta;
    }
  }

  // 2. Tiled Matrix Multiplication (C += alpha * A * B)
  // Parallelize the outermost loop (distribute tiles to threads)
  #pragma omp parallel for private(jj, kk, i, j, k)
  for (ii = 0; ii < NI; ii += BS) //* Increamtning the loop by 32, cause we want to do computation is 32 size chunks to have optimal speed
  {

    //* Iterate through tiles of Column J
    for (jj = 0; jj < NJ; jj += BS)

      //* Iterate through tiles of Common Dimension K
      for (kk = 0; kk < NK; kk += BS)
      {

        //* --- INNER LOOPS (Process ONE 32x32 Tile) ---

        //* Loop 'i': Rows of the tile
        for (i = ii; i < min(ii + BS, NI); i++)
        {

          // Loop 'k': Columns of A / Rows of B
          // REORDERED! Putting 'k' here is better for cache than 'j'
          for (k = kk; k < min(kk + BS, NK); k++)
          {

            // Optimization: Load A value once, keep in register
            float val_A = alpha * A[i * NK + k];

          // Loop 'j': Columns of the tile
          // This is Stride-1 (Sequential) access for C and B -> Great for Vectorization
#pragma omp simd //* Vectorization applies math simultaneously all at once to compute faster
            for (j = jj; j < min(jj + BS, NJ); j++)
            {
              C[i * NJ + j] += val_A * B[k * NJ + j];
            }
          }
        }
      }
  }
}

//* 2D Tiling 
static void kernel_gemm2(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{
  int i, j, k;

  // => Form C := alpha*A*B + beta*C,
  // A is NIxNK
  // B is NKxNJ
  // C is NIxNJ

  // Outer loops: Iterate over tiles
  for (ii = 0; ii < NI; ii += BS)
    for (jj = 0; jj < NJ; jj += BS)
      // Inner loops: Iterate inside tile
      for (i = ii; i < min(ii + BS, NI); i++)
        for (j = jj; j < min(jj + BS, NJ); j++)
          // K loop runs FULL length (Not Tiled)
          for (k = 0; k < NK; k++)
            C[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
}

//* 3D Tiling 
static void kernel_gemm3(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
{
  int i, j, k;
  int ii, jj, kk; // Declare outer loop counters

  // --- Step 1: Scale C by Beta ---
  // You must do this separately or first, otherwise C loses its initial value contribution.
  // Use simple loops here for clarity (compiler optimizes this easily).
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] *= beta;
    }
  }

  // --- Step 2: Tiled Multiplication ---
  for (ii = 0; ii < NI; ii += BS) {
    for (jj = 0; jj < NJ; jj += BS) {
      for (kk = 0; kk < NK; kk += BS) {

        // Optimized Inner Loops: i-k-j Order
        for (i = ii; i < min(ii + BS, NI); i++) {
          
          // ERROR WAS HERE: Added "{" to start the K block
          for (k = kk; k < min(kk + BS, NK); k++) {
            
            // Optimization: Pre-load A into a register
            float A_val = alpha * A[i * NK + k];

            // J loop MUST be inside K loop to multiply A_val against the row of B
            for (j = jj; j < min(jj + BS, NJ); j++) {
              C[i * NJ + j] += A_val * B[k * NJ + j];
            }
          } // Added "}" to close the K block
        }
      }
    }
  }
}



// static void kernel_gemm_simd(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
// {
//   int i, j, k;
//   int ii, jj, kk;

//   // 1. Beta Scaling (Standard Parallel Implementation)
//   // You could vectorize this too, but the main gain is in the matrix multiplication
//   #pragma omp parallel for private(i, j)
//   for (i = 0; i < NI; i++) {
//     for (j = 0; j < NJ; j++) {
//       C[i * NJ + j] *= beta;
//     }
//   }

//   // 2. Tiled Matrix Multiplication with AVX2 Intrinsics
//   // Parallelize outer loops
//   #pragma omp parallel for private(jj, kk, i, j, k)
//   for (ii = 0; ii < NI; ii += BS) {
//     for (jj = 0; jj < NJ; jj += BS) {
//       for (kk = 0; kk < NK; kk += BS) {

//         // --- Inner Tile Loops ---
//         for (i = ii; i < min(ii + BS, NI); i++) {
          
//           for (k = kk; k < min(kk + BS, NK); k++) {
            
//             // Broadcast the value of A into a 256-bit register (8 copies of A_val)
//             // A_val is "scalar" for this row, so we fill a vector with it.
//             __m256 vec_A = _mm256_set1_ps(alpha * A[i * NK + k]);

//             // Vectorized J Loop: Process 8 floats at a time
//             // We increment j by 8 because we handle 8 elements per iteration
//             for (j = jj; j < min(jj + BS, NJ); j += 8) {
              
//               // 1. Load 8 floats from C into a register
//               __m256 vec_C = _mm256_loadu_ps(&C[i * NJ + j]);

//               // 2. Load 8 floats from B into a register
//               __m256 vec_B = _mm256_loadu_ps(&B[k * NJ + j]);

//               // 3. Fused Multiply-Add: vec_C = (vec_A * vec_B) + vec_C
//               // This is faster than separate mul and add instructions
//               vec_C = _mm256_fmadd_ps(vec_A, vec_B, vec_C);

//               // 4. Store the result back to C array
//               _mm256_storeu_ps(&C[i * NJ + j], vec_C);
//             }
//           }
//         }
//       }
//     }
//   }
// }

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

  /* Run kernel. */
  kernel_gemm(C, A, B, 1.5, 2.5);

  /* Stop and print timer. */
  toc(&timer, "kernel execution");

  /* Print results. */
  print_array_sum(C);
  printf("OpenMP will use up to %d threads\n", omp_get_max_threads());
  // printf("Running with %d threads\n", omp_get_max_threads());

  /* free memory for A, B, C */
  free(A);
  free(B);
  free(C);

  return 0;
}




