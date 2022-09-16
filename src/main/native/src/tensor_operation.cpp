#include "tensor_operation.h"

#include "stdio.h"

#include "cuerror.h"
#include "kernel_traits.h"
#include "settings.h"

#include "gpu_utility.h"

#include "solver_kernel.cuh"

#include "quda.h"

///
template void TensorOperation::gatherVector<double>(int nnz, cudaDataType valueType, double *PT1, int *PT2, double *PT);

TensorOperation::TensorOperation(cudaStream_t stream) : stream(stream), pBufferSizeInBytes(0), Psize(0), PT_nnz(0) {

    cusparseStatus_t status;

    /// cuSparse computation context
    status = cusparseCreate(&sparseHandle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        auto errorName = cusparseGetErrorName(status);
        auto errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[gpu/cusparse] cusparse initialization failure ; ( %s ) %s \n", errorName, errorStr);
        exit(1);
    }
    status = cusparseSetStream(sparseHandle, stream);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        auto errorName = cusparseGetErrorName(status);
        auto errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[gpu/cusparse] cusparse stream failure ; ( %s ) %s \n", errorName, errorStr);
        exit(1);
    }

    cublasStatus_t cublasStatus;

    /// cuBlas computation context
    cublasStatus = cublasCreate(&cublasHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        auto errorName = cusparseGetErrorName(status);
        auto errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[gpu/cusparse] cublas handle failure ; ( %s ) %s \n", errorName, errorStr);
        exit(1);
    }

    cublasStatus = cublasSetStream(cublasHandle, stream);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        auto errorName = cusparseGetErrorName(status);
        auto errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[gpu/cublas] stream not set for cublas handler; ( %s ) %s \n", errorName, errorStr);
        exit(1);
    }
};

void TensorOperation::vectorNorm(int n, double *x, double *result) {

    checkCublasStatus(cublasDnrm2(cublasHandle, n, x, 1, result));

    if (settings::get()->DEBUG_SOLVER_CONVERGENCE) {
        checkCudaStatus(cudaStreamSynchronize(stream));
        printf("[cublas.norm] constraint evalutated norm  = %e \n", *result);
    } else {
        // blad na cublas RESULT jest lokalny  zawsze
        /// checkCublasStatus(cublasDnrm2(cublasHandle, n, x, 1, result))
    }
}

void TensorOperation::cublasAPIDaxpy(int n, const double *alpha, const double *x, int incx, double *y, int incy) {

    checkCublasStatus(::cublasDaxpy(cublasHandle, n, alpha, x, incx, y, incy));

    if (settings::get()->DEBUG_SOLVER_CONVERGENCE) {
        checkCudaStatus(cudaStreamSynchronize(stream));
        printf("[cublas.norm] cublasDaxpy \n");
    }
}

void TensorOperation::memsetD32I(int *devPtr, int value, size_t size, cudaStream_t stream) {

    kernelMemsetD32I(stream, devPtr, value, size);

    if (settings::get()->DEBUG_CHECK_ARG) {
        /// block and wait for execution
        checkCudaStatus(cudaStreamSynchronize(stream));
    }
}

/// <summary>
/// Transform COO storage format to CSR storage format.
/// If initialize is requsted procedure will compute PT (permutation vector) and csrRowPtrA vector.
/// In all cases cooValues will be recomputed with permutation vector.
/// </summary>
/// <param name="m">IN</param>
/// <param name="n">IN</param>
/// <param name="nnz">IN</param>
/// <param name="cooRowInd">IN/OUT sorted out</param>
/// <param name="cooColInd">IN/OUT sorted out  [ csrColIndA ] </param>
/// <param name="csrRowPtrA">OUT vector - m + 1  [ csr compact form ] </param>
/// <param name="PT">IN/OUT vector[nnz], permutation from coo into csr</param>
/// <param name="initialize">if true PT is not reused in computation and csrRowPtrA is build</param>
void TensorOperation::convertToCsr(int m, int n, int nnz, int *cooRowInd, int *cooColInd, int *csrRowInd, int *PT) {

    cusparseStatus_t status;

    /// required minimum vectors lengths
    if (PT1.get_size() < nnz) {
        PT1 = utility::dev_vector<int>{nnz, stream};
        PT2 = utility::dev_vector<int>{nnz, stream};
    }

    // first acc vector identity vector
    checkCusparseStatus(cusparseCreateIdentityPermutation(sparseHandle, nnz, PT1));

    /// seconf acc identity vector
    checkCusparseStatus(cusparseCreateIdentityPermutation(sparseHandle, nnz, PT2));

    /// required computation Buffer
    status = cusparseXcoosort_bufferSizeExt(sparseHandle, m, n, nnz, cooRowInd, cooColInd, &pBufferSizeInBytes);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        auto errorName = cusparseGetErrorName(status);
        auto errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[cusparse/error]  Xcoosort buffer size estimate failes; ( %s ) %s \n", errorName, errorStr);
        exit(1);
    }

    /// async prior action with host reference
    checkCudaStatus(cudaStreamSynchronize(stream));

    if (pBufferSizeInBytes > pBuffer.get_size()) {
        pBuffer = utility::dev_vector<char>(pBufferSizeInBytes, stream);
    }

    checkCudaStatus(cudaStreamSynchronize(stream));

    /// recompute sort by column action on COO tensor
    status = cusparseXcoosortByColumn(sparseHandle, m, n, nnz, cooRowInd, cooColInd, PT1, pBuffer);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        checkCusparseStatus(status);
    }
    /// recompute sort by row action on COO tensor
    status = cusparseXcoosortByRow(sparseHandle, m, n, nnz, cooRowInd, cooColInd, PT2, pBuffer);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        auto errorName = cusparseGetErrorName(status);
        auto errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[cusparse/error]  Xcoosort by row failure ; ( %s ) %s \n", errorName, errorStr);
        exit(1);
    }

    if (settings::get()->DEBUG_CHECK_ARG) {
        checkCudaStatus(cudaStreamSynchronize(stream));
    }
    // prior action :: if the Stream Ordered Memory Allocator ???

    gatherVector<int>(nnz, CUDA_R_32F, PT1, PT2, PT);

    if (settings::get()->DEBUG_COO_FORMAT) {
        checkCudaStatus(cudaStreamSynchronize(stream));
        utility::stdout_vector(utility::dev_vector<int>{cooRowInd, (size_t)nnz, stream}, "cooRowInd --- Xsoort");
        utility::stdout_vector(utility::dev_vector<int>{cooColInd, (size_t)nnz, stream}, "cooColInd --- Xsoort");
    }

    /// create csrRowPtrA    -  async execution
    status = cusparseXcoo2csr(sparseHandle, cooRowInd, nnz, m, csrRowInd, CUSPARSE_INDEX_BASE_ZERO);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        auto errorName = cusparseGetErrorName(status);
        auto errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[cusparse/error]  conversion to csr storage format ; ( %s ) %s \n", errorName, errorStr);
        exit(1);
    }

    checkCudaStatus(cudaStreamSynchronize(stream));

    if (settings::get()->DEBUG_CSR_FORMAT) {
        checkCudaStatus(cudaStreamSynchronize(stream));
        utility::stdout_vector(utility::dev_vector<int>{csrRowInd, (size_t)(m + 1), stream}, "csrRowInd -- CSR result !");
    }

    return;
}

template <typename ValueType> void TensorOperation::gatherVector(int nnz, cudaDataType valueType, ValueType *PT1, int *PT2, ValueType *PT) {

    cusparseStatus_t status;

    cusparseSpVecDescr_t vecX; // P2 , vals PT
    cusparseDnVecDescr_t vecY; // P1

    ValueType *X_values = PT;
    int *X_indicies = PT2;
    ValueType *Y = PT1;

    /// not supported CUDA_R_32I
    checkCusparseStatus(cusparseCreateSpVec(&vecX, nnz, nnz, X_indicies, X_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, valueType));
    /// not supported CUDA_R_32I
    checkCusparseStatus(cusparseCreateDnVec(&vecY, nnz, Y, valueType));

    /// permutation vector compact operatior PT[u] = PT1[PT2[u]]

    /// OPERATION ::  X_values[i] = Y[X_indices[i]]
    status = cusparseGather(sparseHandle, vecY, vecX);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        log_error(status, "[cusparse] gather operation failed !");
        exit(1);
    }

    checkCusparseStatus(cusparseDestroyDnVec(vecY));
    checkCusparseStatus(cusparseDestroySpVec(vecX));
}

void TensorOperation::invertPermuts(int n, int *PT, int *INV) {

    inversePermutationVector(stream, PT, INV, n);

    if (settings::get()->DEBUG_CHECK_ARG) {
        /// block and wait for execution
        checkCudaStatus(cudaStreamSynchronize(stream));
    }
}


void TensorOperation::stdout_coo_tensor(cudaStream_t stream, int m, int n, int nnz, int *d_cooRowInd, int *d_cooColInd, double *d_cooVal, const char* title) {

    utility::host_vector<int> cooRowInd{(size_t)nnz};
    utility::host_vector<int> cooColInd{(size_t)nnz};
    utility::host_vector<double> cooVal{(size_t)nnz};

    cooRowInd.memcpy_of(utility::dev_vector<int>(d_cooRowInd, nnz, stream), stream);
    cooColInd.memcpy_of(utility::dev_vector<int>(d_cooColInd, nnz, stream), stream);
    cooVal.memcpy_of(utility::dev_vector<double>(d_cooVal, nnz, stream), stream);

    cudaStreamSynchronize(stream);

    utility::infoLog( "\n---- %s\n", title);
    /// Stdout ad m x n format
    int idx = 0;
    for (int T = 0; T < m; ++T) {
        for (int L = 0; L < n; ++L) {
            if (cooRowInd[idx] == T && cooColInd[idx] == L) {
                utility::infoLog(" %8.3f ,", cooVal[idx]);
                ++idx;
            } else {
                utility::infoLog(" %8.3f ,", 0.0);
            }
        }
        utility::infoLog(" \n ");
    }
    utility::infoLog("\n");
}


void TensorOperation::stdout_coo_vector(cudaStream_t stream, int m, int n, int nnz, int *d_cooRowInd, int *d_cooColInd, double *d_cooVal,const char* title) {

    utility::host_vector<int> cooRowInd{(size_t)nnz};
    utility::host_vector<int> cooColInd{(size_t)nnz};
    utility::host_vector<double> cooVal{(size_t)nnz};

    cooRowInd.memcpy_of(utility::dev_vector<int>(d_cooRowInd, nnz, stream), stream);
    cooColInd.memcpy_of(utility::dev_vector<int>(d_cooColInd, nnz, stream), stream);
    cooVal.memcpy_of(utility::dev_vector<double>(d_cooVal, nnz, stream), stream);

    cudaStreamSynchronize(stream);

    utility::infoLog("\n--- %s \n", title);
    /// all indicies withe value in coo format
    for (int T = 0; T < nnz; ++T) {
        utility::infoLog(" %d , %d %d  -  %7.3f  \n", T, cooRowInd[T], cooColInd[T], cooVal[T]);
    }
    utility::infoLog("\n");
}

TensorOperation::~TensorOperation() {
    //
    checkCusparseStatus(cusparseDestroy(sparseHandle));

    checkCublasStatus(cublasDestroy(cublasHandle));
}


#undef validateStream