#include "tensor_operation.h"

#include "stdio.h"

TensorOperation::TensorOperation(cudaStream_t stream) : stream(stream), pBuffer(NULL), Prm(NULL), Psize(0) {

    status = cusparseCreate(&cuSparseHandle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        const char* errorName = cusparseGetErrorName(status);
        const char* errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[gpu/cusparse] cusparse initialization failure ; ( %s ) %s \n", errorName, errorStr);
        exit(1);
    }
    status = cusparseSetStream(cuSparseHandle, stream);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        const char *errorName = cusparseGetErrorName(status);
        const char *errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[gpu/cusparse] cusparse stream failure ; ( %s ) %s \n", errorName, errorStr);
        exit(1);
    }
};

void TensorOperation::convertToCSR(int *cooRowInd, int *cooColInd, double *cooVal, int nnz, int ld, int *csrRowInd) {
           
    /// Ensure required buffer size
    int buffSize = pBufferSizeInBytes;

    status = cusparseXcoosort_bufferSizeExt(cuSparseHandle, ld, ld, nnz, cooRowInd, cooColInd, &pBufferSizeInBytes);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        const char *errorName = cusparseGetErrorName(status);
        const char *errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[gpu/cusparse] cusparse stream failure ; ( %s ) %s \n", errorName, errorStr);
        exit(1);
    }

    if (pBufferSizeInBytes > buffSize) {

        if (pBuffer != NULL) {
            cudaFreeAsync(pBuffer, stream);
        }
        /// Alllocate temporary buffer
        cudaMallocAsync((void **)&pBuffer, pBufferSizeInBytes, stream);
    }

    if (nnz > Psize) {

        if (Prm != NULL) {

        }
    }



    // # PERMUTATION
    cusparseCreateIdentityPermutation(cuSparseHandle, nnz, Prm);

    /// ONLY ONCE AS PRE-PROCESSING FOR SECOND ROUND

    // d_cooRowInd ?? - first round computed by memcpy !!!
    // d_cooColInd ?? - first round computed by memcpy !!!

    /// ===================================================== ///

    //# PROCEDURE SORT IN ROW  - destructive
    cusparseXcoosortByRow(cuSparseHandle, ld, ld, nnz,
                          cooRowInd, // IN.OUT
                          cooColInd, // IN.OUT
                          Prm,             // IN.OUT
                          pBuffer);

    /// ===================================================== ///

    // # GATHER ELEMETNS
    cusparseDgthr(cuSparseHandle, nnz,
                  cooVal, // IN
                  cooVal, // OUT
                  Prm, CUSPARSE_INDEX_BASE_ZERO);

    /// Inverse all indicies for Direct Layout processing
    //inverse_indices(Prm, Pnvi, nnz);


    // cusparseStatus_t cusparseXcoo2csr(cusparseHandle_t handle, const int *cooRowInd, int nnz, int m, int *csrRowPtr,cusparseIndexBase_t idxBase);


}

TensorOperation::~TensorOperation() {
    //
    cudaError_t cudaError; 
    if (Prm != NULL) {
        cudaError = cudaFreeAsync(Prm, stream);
        if (cudaError != cudaSuccess) {
            const char *errorName = cudaGetErrorName(cudaError);
            const char *errorStr = cudaGetErrorString(cudaError);
            fprintf(stderr, "[gpu/cusparse] async free failed; ( %s ) %s \n", errorName, errorStr);
        }
    }

    if (pBuffer != NULL) {
        cudaFreeAsync(pBuffer, stream);
        if (cudaError != cudaSuccess) {
            const char *errorName = cudaGetErrorName(cudaError);
            const char *errorStr = cudaGetErrorString(cudaError);
            fprintf(stderr, "[gpu/cusparse] async free failed; ( %s ) %s \n", errorName, errorStr);
        }
    }        

    if (cuSparseHandle != NULL) {

        status = cusparseDestroy(cuSparseHandle);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            const char *errorName = cusparseGetErrorName(status);
            const char *errorStr = cusparseGetErrorString(status);
            fprintf(stderr, "[gpu/cusparse] handler destroy failed; ( %s ) %s \n", errorName, errorStr);
            exit(1);
        }
    }
}