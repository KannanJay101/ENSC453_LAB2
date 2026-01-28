#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "my_timer.h"

#define NI 2048
#define NJ 2048
#define NK 2048

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



//! Main computational kernel. The whole function will be timed,  including the call and return. */
// static void kernel_gemm(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta)
// {
//   int i, j, k;

//   // => Form C := alpha*A*B + beta*C,
//   // A is NIxNK
//   // B is NKxNJ
//   // C is NIxNJ
//   // #pragma omp parallel for private(j,k)



 
//   for (i = 0; i < NI; i++)
//   {

//     for (j = 0; j < NJ; j++)
//     {
//       C[i * NJ + j] *= beta;
//     }
//     for (j = 0; j < NJ; j++)
//     {
//       for (k = 0; k < NK; ++k)
//       {
//         C[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
//       }
//     }
//   }
// }

//* Note: This CPU contains -> 6 P-Cores (Fast) , 8 E-Cores (Slower),  6 P-Cores + 8 E-Cores = 20 Threads in Total
//* Tiling (Do mulitplication in small chunks that fit in the cache), therefore the CPU reuses data instead of redoing the data again

#define BS 32 //*Defining new block size 

// Helper macro to ensure we don't go out of bounds
#define min(a,b) (((a)<(b))?(a):(b)) 

static void kernel_gemm(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k;      // Inner loop counters (workers)
  int ii, jj, kk;   // Outer loop counters (tile movers)

  // 1. Handle Beta Scaling (C = C * beta)
  // We do this first so we don't have to multiply by beta inside the critical path
  //* Frist scaling Matrix C by beta and using parallezation to quickly compute the first part of the computation
  //* 
  #pragma omp parallel for private(i, j) //* 
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      
      C[i*NJ+j] *= beta;
    }
  }

  

  // 2. Tiled Matrix Multiplication (C += alpha * A * B)
  // Parallelize the outermost loop (distribute tiles to threads)

  //* Note: private(jj, kk, i, j, k) ->
  #pragma omp parallel for private(jj, kk, i, j, k) 
  //#pragma omp parallel for num_threads(12) private(jj,kk,i,j,k)
  for (ii = 0; ii < NI; ii += BS) {
    
    // Iterate through tiles of Column J
    for (jj = 0; jj < NJ; jj += BS) {
      
      // Iterate through tiles of Common Dimension K
      for (kk = 0; kk < NK; kk += BS) {

        // --- INNER LOOPS (Process ONE 32x32 Tile) ---
        
        // Loop 'i': Rows of the tile
        for (i = ii; i < min(ii + BS, NI); i++) {
          
          // Loop 'k': Columns of A / Rows of B
          // REORDERED! Putting 'k' here is better for cache than 'j'
          for (k = kk; k < min(kk + BS, NK); k++) {
            
            // Optimization: Load A value once, keep in register
            float val_A = alpha * A[i*NK+k];

            // Loop 'j': Columns of the tile
            // This is Stride-1 (Sequential) access for C and B -> Great for Vectorization
            #pragma omp simd 
            for (j = jj; j < min(jj + BS, NJ); j++) {
              C[i*NJ+j] += val_A * B[k*NJ+j];
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
