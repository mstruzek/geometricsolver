#ifndef _CUERROR_H_
#define _CUERROR_H_

#include "cublas_v2.h"
#include "cuda_runtime_api.h"
#include "cusolverDn.h"


/// cuBLAS - error handler
#define checkCublasStatus(status) error::_checkCublasStatus(status, __LINE__, __FILE__)

/// CUDA - error handler
#define checkCudaStatus(status) error::_checkCudaStatus(status, __LINE__, __FILE__)

/// cuSolver - error handler
#define checkCuSolverStatus(status) error::_checkCuSolverStatus(status, __LINE__, __FILE__)


namespace error
{
		/// implementations

void _checkCublasStatus(cublasStatus_t status, size_t __line, const char *__file__);

void _checkCudaStatus(cudaError_t status, size_t __line, const char *__file__);

void _checkCuSolverStatus(cusolverStatus_t status, size_t _line_, const char *__file__);

}




#endif // _CUERROR_H_