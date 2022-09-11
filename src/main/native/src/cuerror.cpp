#include "cuerror.h"

#define _CRT_SECURE_NO_WARNINGS

#include <stdexcept>
#include <sstream>
#include <string>

#include "gpu_utility.h"

namespace cuda_error {

void _checkCudaStatus(cudaError_t status, size_t __line, const char *__fileName) {
    if (status != cudaSuccess) {
        std::stringstream ss;
        ss << "[cuda/error] cuda API failed with status ; ";
        ss << "( " << cudaGetErrorName(status) << " ) " << cudaGetErrorString(status) << "#" << __line << " "
           << __fileName << std::endl;
        std::string message = ss.str();
        fprintf(stderr, message.c_str());
        throw std::logic_error(message);
    }
}

void _checkCusparseStatus(cusparseStatus_t status, size_t __line, const char *__fileName) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "[cuSparse/error ] cuSparse API failed with status ; ";
        ss << "( " << cusparseGetErrorName(status) << " ) " << cusparseGetErrorString(status) << "#" << __line << " "
           << __fileName << std::endl;
        std::string message = ss.str();
        fprintf(stderr, message.c_str());
        throw std::logic_error(message);
    }
}

void _checkCuSolverStatus(cusolverStatus_t status, size_t __line, const char *__fileName) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "[cuSolver/error] CuSolver API failed with status ; ";
        ss << "( " << cusolverGetErrorName(status) << " ) "
           << "#" << __line << " " << __fileName << std::endl;
        std::string message = ss.str();
        fprintf(stderr, message.c_str());
        throw std::logic_error(message);
    }
}

void _checkCublasStatus(cublasStatus_t status, size_t __line, const char *__fileName) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "[cuBLAS/error] cuBLAS API failed with status  ; ";
        ss << "( " << cublasGetStatusName(status) << " ) " << cublasGetStatusString(status) << "#" << __line << " "
           << __fileName << std::endl;
        std::string message = ss.str();
        fprintf(stderr, message.c_str());
        throw std::logic_error(message);
    }
}

} // namespace cuda_error