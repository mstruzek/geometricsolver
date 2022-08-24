#ifndef _CUERROR_H_
#define _CUERROR_H_

#include "cublas_v2.h"
#include "cuda_runtime_api.h"
#include "cusolverDn.h"



#define checkCublasStatus(status) errors::_checkCublasStatus(status, __LINE__, __FILE__)
#define checkCudaStatus(status) errors::_checkCudaStatus(status, __LINE__, __FILE__)
#define checkCuSolverStatus(status) errors::_checkCuSolverStatus(status, __LINE__, __FILE__)


namespace errors
{

void _checkCublasStatus(cublasStatus_t status, size_t __line, const char *__file__);

void _checkCudaStatus(cudaError_t status, size_t __line, const char *__file__);

void _checkCuSolverStatus(cusolverStatus_t status, size_t _line_, const char *__file__);
}




#endif // _CUERROR_H_
