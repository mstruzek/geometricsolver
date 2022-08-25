#include "cuerror.h"

#include <stdexcept>

namespace error
{

void _checkCudaStatus(cudaError_t status, size_t __line__, const char *__file__)
{
    if (status != cudaSuccess)
    {
        const char *errorName = cudaGetErrorName(status);
        const char *errorStr = cudaGetErrorString(status);
        printf("[ cuda / error ] %s#%d : cuda API failed (%d),  %s  : %s \n", __file__, (int)__line__, status,
               errorName, errorStr);
        throw std::logic_error("cuda API error");
    }
}

void _checkCuSolverStatus(cusolverStatus_t status, size_t __line__, const char *__file__)
{
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        printf("[ CuSolver / error ] %s#%d : CuSolver API failed with status %d \n", __file__, (int)__line__, status);
        throw std::logic_error("CuSolver error");
    }
}

void _checkCublasStatus(cublasStatus_t status, size_t __line__, const char *__file__)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        const char *statusName = cublasGetStatusName(status);
        const char *statusString = cublasGetStatusString(status);
        printf("[ cuBLAS / error ] %s#%d : cuBLAS API failed with status (%d) , %s : %s \n", __file__, (int)__line__,
               status, statusName, statusString);
        throw std::logic_error("cuBLAS error");
    }
}

} // namespace errors