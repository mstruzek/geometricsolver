#ifndef _LINEAR_SYSTEM_CUH_
#define _LINEAR_SYSTEM_CUH_

#include <stdexcept>

#include "cublas_v2.h"
#include "cuda_runtime_api.h"
#include "cusolverDn.h"

#define checkCublasStatus(status)               errors::_checkCublasStatus(status, __LINE__, __FILE__)
#define checkCudaStatus(status)                 errors::_checkCudaStatus(status, __LINE__, __FILE__)
#define checkCuSolverStatus(status)             errors::_checkCuSolverStatus(status, __LINE__, __FILE__)


namespace errors {

    void _checkCublasStatus(cublasStatus_t status, size_t __line, const char * __file__);

    void _checkCudaStatus(cudaError_t status, size_t __line, const char * __file__);

    void _checkCuSolverStatus(cusolverStatus_t status, size_t _line_, const char* __file__);
}


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