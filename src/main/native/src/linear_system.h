#ifndef _LINEAR_SYSTEM_CUH_
#define _LINEAR_SYSTEM_CUH_

#include <stdexcept>

#include "cublas_v2.h"
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#define checkCublasStatus(status)               errors::_checkCublasStatus(status, __LINE__)
#define checkCudaStatus(status)                 errors::_checkCudaStatus(status, __LINE__)
#define checkCuSolverStatus(status)             errors::_checkCuSolverStatus(status, __LINE__)


namespace errors {

    void _checkCublasStatus(cublasStatus_t status, size_t __line);

    void _checkCudaStatus(cudaError_t status, size_t __line);

    void _checkCuSolverStatus(cusolverStatus_t status, size_t _line_);
}


/// <summary>
/// linear system solver from cuSolver library : method ( cusolverDnDpotrs ) with factorization ( cusolverDnDpotrf )
/// </summary>
/// <param name="A">input</param>
/// <param name="b">input/output vector</param>
/// <param name="N"></param>
void linear_system_method_0(double *A, double *b, size_t N);


/// <summary>
/// rest cuSolver additional state data
/// </summary>
void linear_system_method_0_reset();


#endif //_LINEAR_SYSTEM_CUH_