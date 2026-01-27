#include <stdio.h>
#include <stdlib.h>
#include<omp.h>
#include "my_timer.h"

#define NI 2048
#define NJ 2048
#define NK 2048



/* Array initialization. */
static
void init_array(float C[NI*NJ], float A[NI*NK], float B[NK*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i*NJ+j] = (float)((i*j+1) % NI) / NI;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i*NK+j] = (float)(i*(j+1) % NK) / NK;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i*NJ+j] = (float)(i*(j+2) % NJ) / NJ;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static

void print_array(float C[NI*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      printf("C[%d][%d] = %f\n", i, j, C[i*NJ+j]);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array_sum(float C[NI*NJ])
{
  int i, j;

  float sum = 0.0;
  
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      sum += C[i*NJ+j];

  printf("sum of C array = %f\n", sum);
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;
  int tile_sz = 16;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ

/*
// Tiling of the i and j loops
  for (i = 0; i < NI; i += tile_sz) {
    for (j = 0; j < NJ; j += tile_sz) {

      for (ii = i; ii < i + tile_sz; ii++) {
        for (jj = j; jj < j + tile_sz; jj++) {
          C[ii*NJ + jj] *= beta;
        }
      }

      for (k = 0; k < NK; k++) {
        for (ii = i; ii < i + tile_sz; ii++) {
          for (jj = j; jj < j + tile_sz; jj++) {
            C[ii*NJ + jj] += alpha * A[ii*NK + k] * B[k*NJ + jj];
          }
        }
      }

    }
  }
*/


/*
// Tiling of the i and k loops
  for (i = 0; i < NI; i += tile_sz) {
    for (j = 0; j < NJ; j++) {

      for (ii = i; ii < i + tile_sz; ii++) {
        C[ii*NJ + j] *= beta;
      }

      for (k = 0; k < NK; k += tile_sz) {

        for (ii = i; ii < i + tile_sz; ii++) {
          for (kk = k; kk < k + tile_sz; kk++) {
            C[ii*NJ + j] += alpha * A[ii*NK + kk] * B[kk*NJ + j];
          }
        }
      }
    }
  }
*/
 
/*
//Tiling of j and k loops
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j += tile_sz) {

      for (jj = j; jj < j + tile_sz; jj++) {
        C[i*NJ + jj] *= beta;
      }

      for (k = 0; k < NK; k += tile_sz) {

        for (jj = j; jj < j + tile_sz; jj++) {
          for (kk = k; kk < k + tile_sz; kk++) {
            C[i*NJ + jj] += alpha * A[i*NK + kk] * B[kk*NJ + jj];
          }
        }
      }

    }
  }
*/

//Tiling of all three (i, j, k) loops
  for (i = 0; i < NI; i += tile_sz) {
    for (j = 0; j < NJ; j += tile_sz) {

      for (ii = i; ii < i + tile_sz; ii++) {
        for (jj = j; jj < j + tile_sz; jj++) {
          C[ii*NJ + jj] *= beta;
        }
      }

      for (k = 0; k < NK; k += tile_sz) {

        for (ii = i; ii < i + tile_sz; ii++) {
          for (jj = j; jj < j + tile_sz; jj++) {
            for (kk = k; kk < k + tile_sz; kk++) {
              C[ii*NJ + jj] += alpha * A[ii*NK + kk] * B[kk*NJ + jj];
            }
          }
        }
      }

    }
  }


}


int main(int argc, char** argv)
{

  
  /* Variable declaration/allocation. */
  float *A = (float *)malloc(NI*NK*sizeof(float));
  float *B = (float *)malloc(NK*NJ*sizeof(float));
  float *C = (float *)malloc(NI*NJ*sizeof(float));

  /* Initialize array(s). */
  init_array (C, A, B);

  /* Start timer. */
  timespec timer = tic();

  /* Run kernel. */
  kernel_gemm (C, A, B, 1.5, 2.5);

  /* Stop and print timer. */
  toc(&timer, "kernel execution");
  
  /* Print results. */
  print_array_sum (C);
  printf("OpenMP will use up to %d threads\n", omp_get_max_threads());
  //printf("Running with %d threads\n", omp_get_max_threads());

  /* free memory for A, B, C */
  free(A);
  free(B);
  free(C);
  
  return 0;
}
