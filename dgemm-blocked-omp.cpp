#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <omp.h>
#include "likwid-stuff.h"

const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
	 std::vector<double> buf(6 * n * n);
         double* copyA = buf.data() + 0;
         double* copyB = copyA + n * n;
         double* copyC = copyB + n * n;
         
	 memcpy((void *)copyA, (const void *)A, sizeof(double)*n*n);
         memcpy((void *)copyB, (const void *)B, sizeof(double)*n*n);
         memcpy((void *)copyC, (const void *)C, sizeof(double)*n*n);
   	
	 LIKWID_MARKER_INIT;
	#pragma omp parallel
	{
   	   LIKWID_MARKER_REGISTER(MY_MARKER_REGION_NAME);
	}
	#pragma omp parallel
	{
	   LIKWID_MARKER_START(MY_MARKER_REGION_NAME);
           #pragma omp for
	   for(int i = 0; i < n; i += block_size)
            {
                    for(int j = 0; j < n; j += block_size)
                    {
                            for(int k = 0; k < n; k += block_size)
                            {
                                    for(int x = i; x < i + block_size; x++)
                                    {
                                            for(int y = j; y < j + block_size; y++)
                                            {
                                                    for(int z = k; z < k + block_size; z++)
                                                    {
                                                           copyC[x + y * n] += copyA[x + z * n] * copyB[z + y * n];

                                                    }
                                            }

                                    }
                            }

                    }
            }
	   
	   memcpy((void *)C, (const void *)copyC, sizeof(double)*n*n);
	   LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
	}
   LIKWID_MARKER_CLOSE;
   
   // insert your code here: implementation of blocked matrix multiply with copy optimization and OpenMP parallelism enabled

   // be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside the block of parallel code,
   // but before your matrix multiply code, and then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
   // after the matrix multiply code but before the end of the parallel code block.

   std::cout << "Insert your blocked matrix multiply with copy optimization, openmp-parallel edition here " << std::endl;
}
