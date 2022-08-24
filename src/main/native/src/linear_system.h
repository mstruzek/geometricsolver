#ifndef _LINEAR_SYSTEM_CUH_
#define _LINEAR_SYSTEM_CUH_


#include "cuda_runtime_api.h"


/// <summary>
/// linear system solver from cuSolver library : method ( cusolverDnDpotrs ) with factorization ( cusolverDnDpotrf )
/// </summary>
/// <param name="A">input</param>
/// <param name="b">input/output vector</param>
/// <param name="N"></param>
void linear_system_method_cuSolver(double *A, double *b, size_t N, cudaStream_t stream);


/// <summary>
/// rest cuSolver additional state data
/// </summary>
void linear_system_method_cuSolver_reset(cudaStream_t stream);




/// <summary>
/// test vector norm async with result stored on the device
/// </summary>
/// <param name="n"></param>
/// <param name="x"></param>
/// <param name="result"></param>
void linear_system_method_cuBlas_vectorNorm(int n, double *x, double *result, cudaStream_t stream);




#endif //_LINEAR_SYSTEM_CUH_