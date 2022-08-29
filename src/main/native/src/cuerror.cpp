#include "cuerror.h"

#include <stdexcept>
#include <string.h>
namespace cuda_error
{

void _checkCudaStatus(cudaError_t status, size_t __line__, const char *__file__)
{
    if (status != cudaSuccess)
    {
        const char *errorName = cudaGetErrorName(status);
        const char *errorStr = cudaGetErrorString(status);
        const char *format = "[ Cuda / error ] %s#%d : cuda API failed (%d),  %s  : %s \n";
        size_t len = strlen(errorName) + strlen(errorStr) + strlen(format) + strlen(__file__) + 12;
        std::string message(len, ' ');
        sprintf(message.data(), format,  __file__, (int)__line__, status, errorName, errorStr);
        printf(message.data());
        throw std::logic_error(message);
    }
}

void _checkCuSolverStatus(cusolverStatus_t status, size_t __line__, const char *__file__)
{
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        const char *format = "[ CuSolver / error ] %s#%d : CuSolver API failed with status %d \n";
        size_t len = strlen(format) + strlen(__file__) + 12;
        std::string message(len, ' ');
        sprintf(message.data(), format, __file__, (int)__line__, status);
        printf(message.data());
        throw std::logic_error(message);
    }
}

void _checkCublasStatus(cublasStatus_t status, size_t __line__, const char *__file__)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        const char *statusName = cublasGetStatusName(status);
        const char *statusString = cublasGetStatusString(status);
        const char *format = "[ cuBLAS / error ] %s#%d : cuBLAS API failed with status (%d) , %s : %s \n";
        size_t len = strlen(format) + strlen(statusName) + strlen(statusString) + strlen(__file__) + 12;
        std::string message(len, ' ');
        sprintf(message.data(), format, __file__, (int)__line__, status, statusName, statusString);
        printf(message.data());
        throw std::logic_error(message);
    }
}

} // namespace errors