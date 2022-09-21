#include "gpu_solver_sparse_ilu02.h"

#include "../cuerror.h"
#include "../gpu_utility.h"
#include "../quda.h"
#include "../settings.h"

///
/// "warning C4996: 'cusparseCreateCsrsv2Info': please use cusparseSpSV instead"
///
#pragma warning(disable : 4996)

#ifdef DEBUG_GPU
#define validateStream validateStreamState()
#else
#define validateStream
#endif

namespace solver {

#define CSR(d_csrVal, d_csrRowPtr, d_csrColInd) d_csrVal, d_csrRowPtr, d_csrColInd

#define HANDLE_STATUS status =
GPUSolverSparseILU02::GPUSolverSparseILU02(cudaStream_t stream) : stream(stream) {
    cusparseStatus_t status;
    HANDLE_STATUS cusparseCreate(&handle);
    if (!status) {
        fprintf(stderr, "[solver.ilu02] error solver not initialized; %s . %s \n", cusparseGetErrorName(status), cusparseGetErrorString(status));
        exit(-1);
    }

    HANDLE_STATUS cusparseSetStream(handle, stream);
    if (!status) {
        fprintf(stderr, "[solver.ilu02] error stream not associated;  %s . %s \n", cusparseGetErrorName(status), cusparseGetErrorString(status));
        exit(-1);
    }
}

void GPUSolverSparseILU02::configure(int parameterId, int valueInt, double valueDouble) {
    ///
}

void GPUSolverSparseILU02::setupSparseMatDescriptors() {
    // *************************************************************************** //
    // step 1: create a descriptor which contains
    // - matrix M is base-1
    // - matrix L is base-1
    // - matrix L is lower triangular
    // - matrix L has unit diagonal
    // - matrix U is base-1
    // - matrix U is upper triangular
    // - matrix U has non-unit diagonal
    // *************************************************************************** //
    checkCusparseStatus(cusparseCreateMatDescr(&descr_M));
    checkCusparseStatus(cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO));
    checkCusparseStatus(cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL));
    // *************************************************************************** //
    checkCusparseStatus(cusparseCreateMatDescr(&descr_L));
    checkCusparseStatus(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO));
    checkCusparseStatus(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCusparseStatus(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
    checkCusparseStatus(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT));
    // *************************************************************************** //
    checkCusparseStatus(cusparseCreateMatDescr(&descr_U));
    checkCusparseStatus(cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO));
    checkCusparseStatus(cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCusparseStatus(cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER));
    checkCusparseStatus(cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT));
    // *************************************************************************** //
}

void GPUSolverSparseILU02::setupSolverInfoStructures() {
    // *************************************************************************** //
    // step 2: create a empty info structure
    // we need one info for csrilu02 and two info's for csrsv2
    checkCusparseStatus(cusparseCreateCsrilu02Info(&info_M));
    checkCusparseStatus(cusparseCreateCsrsv2Info(&info_L));
    checkCusparseStatus(cusparseCreateCsrsv2Info(&info_U));
    // *************************************************************************** //
}



void GPUSolverSparseILU02::requestEnsurePBufferSize(int m, int n, int nnz, int *d_csrRowPtr, int *d_csrColInd, double *d_csrVal) {

    cusparseStatus_t status;
    // *************************************************************************** //

    HANDLE_STATUS cusparseDcsrilu02_bufferSize(handle, m, nnz, descr_M, CSR(d_csrVal, d_csrRowPtr, d_csrColInd), info_M, &pBufferSize_M);
    if (!status) {
        fprintf(stderr, "[solver.ilu02] matrix M buffer size request failure ; %s . %s \n", cusparseGetErrorName(status), cusparseGetErrorString(status));
        exit(-1);
    }
    // *************************************************************************** //
    HANDLE_STATUS cusparseDcsrsv2_bufferSize(handle, trans_L, m, nnz, descr_L, CSR(d_csrVal, d_csrRowPtr, d_csrColInd), info_L, &pBufferSize_L);
    if (!status) {
        fprintf(stderr, "[solver.ilu02] matrix L buffer size request failure ; %s . %s \n", cusparseGetErrorName(status), cusparseGetErrorString(status));
        exit(-1);
    }
    // *************************************************************************** //
    HANDLE_STATUS cusparseDcsrsv2_bufferSize(handle, trans_U, m, nnz, descr_U, CSR(d_csrVal, d_csrRowPtr, d_csrColInd), info_U, &pBufferSize_U);
    if (!status) {
        fprintf(stderr, "[solver.ilu02] matrix U buffer size request failure ; %s . %s \n", cusparseGetErrorName(status), cusparseGetErrorString(status));
        exit(-1);
    }
    // *************************************************************************** //

    /// shared buffer between routines
    pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));

    /// increase buffer capacity
    if (pBufferSize > pBuffer.get_size()) {
        pBuffer = utility::dev_vector<char>{pBufferSize};
    }
}

void GPUSolverSparseILU02::performAnalysisIncompleteLU(int m, int n, int nnz, int *d_csrRowPtr, int *d_csrColInd, double *d_csrVal, int *singularity) {

    cusparseStatus_t status;
    // *************************************************************************** //
    // step 4: perform analysis of incomplete LU on M
    //         perform analysis of triangular solve on L
    //         perform analysis of triangular solve on U
    // The lower(upper) triangular part of M has the same sparsity pattern as L(U),
    // we can do analysis of csrilu0 and csrsv2 simultaneously.

    HANDLE_STATUS cusparseDcsrilu02_analysis(handle, m, nnz, descr_M, CSR(d_csrVal, d_csrRowPtr, d_csrColInd), info_M, policy_M, pBuffer);
    if (!status) {
        fprintf(stderr, "[solver.ilu02] matrix M analysis failure ; %s . %s \n", cusparseGetErrorName(status), cusparseGetErrorString(status));
        exit(-1);
    }

    HANDLE_STATUS cusparseXcsrilu02_zeroPivot(handle, info_M, &structural_zero);
    if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
        fprintf(stderr, "A(%d,%d) is missing\n", structural_zero, structural_zero); /// Structural Zero - means no elements in CSR on diagonal !
        *singularity = structural_zero;
        return;
    }

    // *************************************************************************** //

    HANDLE_STATUS cusparseDcsrsv2_analysis(handle, trans_L, m, nnz, descr_L, CSR(d_csrVal, d_csrRowPtr, d_csrColInd), info_L, policy_L, pBuffer);
    if (!status) {
        fprintf(stderr, "[solver.ilu02] matrix L analysis failure ; %s . %s \n", cusparseGetErrorName(status), cusparseGetErrorString(status));
        exit(-1);
    }

    // *************************************************************************** //

    HANDLE_STATUS cusparseDcsrsv2_analysis(handle, trans_U, m, nnz, descr_U, CSR(d_csrVal, d_csrRowPtr, d_csrColInd), info_U, policy_U, pBuffer);
    if (!status) {
        fprintf(stderr, "[solver.ilu02] matrix U analysis failure ; %s . %s \n", cusparseGetErrorName(status), cusparseGetErrorString(status));
        exit(-1);
    }

    // *************************************************************************** //
}

void GPUSolverSparseILU02::tensorFactorization(int m, int n, int nnz, int *csrRowPtr, int *csrColInd, double *csrVal, double *b, double *x,
                                                          int *singularity) {
    cusparseStatus_t status;
    // *************************************************************************** //
    if (descr_M == NULL) {
        /// step 1: create a descriptor for M, L, U
        setupSparseMatDescriptors();

        /// step 2: create a empty info structure
        setupSolverInfoStructures();
    }

    // *************************************************************************** //
    /// step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
    requestEnsurePBufferSize(m, n, nnz, CSR(csrRowPtr, csrColInd, csrVal));

    /// step 3.1: ensure temporary vector Z equal to dimension size
    if (m > d_z.get_size()) {
        d_z = utility::dev_vector<double>{m};
    }

    // *************************************************************************** //
    /// step 4: perform analysis of incomplete LU on M
    performAnalysisIncompleteLU(m, n, nnz, CSR(csrRowPtr, csrColInd, csrVal), singularity);
    if (*singularity > 0) {
        fprintf(stderr, "[solver.ilu02] matrix A is not invertible \n");
        return;
    }

    // *************************************************************************** //
    int enable_boost = 1;
    double tol = 1e-10;
    double boost_val = 1e-10;

    checkCusparseStatus(cusparseDcsrilu02_numericBoost(handle, info_M, enable_boost, &tol, &boost_val));

    // *************************************************************************** //
    /// step 5: M = L * U
    HANDLE_STATUS cusparseDcsrilu02(handle, m, nnz, descr_M, CSR(csrVal, csrRowPtr, csrColInd), info_M, policy_M, pBuffer);
    validateStream;

    if (!status) {
        fprintf(stderr, "[solver.ilu02] Cholesky factorization failure ; %s . %s \n", cusparseGetErrorName(status), cusparseGetErrorString(status));
        exit(-1);
    }

    /// POLICY - no level information ?? ( SYNCHRONIZED request )
    HANDLE_STATUS cusparseXcsrilu02_zeroPivot(handle, info_M, &num_zero);
    if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
        fprintf(stderr, "[solver.ilu02]  U(%d,%d) is zero  - matrix A is not invertible ( advice : numeric boost )\n", num_zero, num_zero);
        *singularity = num_zero;
        return;
    }

    // *************************************************************************** //
    enable_boost = 0;
    checkCusparseStatus(cusparseDcsrilu02_numericBoost(handle, info_M, enable_boost, &tol, &boost_val));
}


void GPUSolverSparseILU02::solveSystem(int m, int n, int nnz, int *csrRowPtr, int *csrColInd, double *csrVal, double *b, double *x, int *singularity) {

    cusparseStatus_t status;

    // *************************************************************************** //
    /// step 6: solve L*z = b
    HANDLE_STATUS cusparseDcsrsv2_solve(handle, trans_L, m, nnz, &alpha, descr_L, CSR(csrVal, csrRowPtr, csrColInd), info_L, b, d_z, policy_L, pBuffer);
    validateStream;

    if (!status) {
        auto errorname = cusparseGetErrorName(status);
        auto description = cusparseGetErrorString(status);
        fprintf(stderr, "[solver.ilu02] matrix U analysis failure ; %s . %s \n", errorname, description);
        exit(-1);
    }

    // *************************************************************************** //
    /// step 7: solve U*x = z
    HANDLE_STATUS cusparseDcsrsv2_solve(handle, trans_U, m, nnz, &alpha, descr_U, CSR(csrVal, csrRowPtr, csrColInd), info_U, d_z, x, policy_U, pBuffer);
    validateStream;

    if (!status) {
        fprintf(stderr, "[solver.ilu02] matrix U analysis failure ; %s . %s \n", cusparseGetErrorName(status), cusparseGetErrorString(status));
        exit(-1);
    }

    // *************************************************************************** //
    *singularity = -1;
    if (settings::DEBUG) {
        fprintf(stderr, "[solver.ILU02] solver routine completed !");
    }

    // *************************************************************************** //
}

GPUSolverSparseILU02::~GPUSolverSparseILU02() {

    if (descr_M) {
        checkCusparseStatus(cusparseDestroyCsrilu02Info(info_M));

        checkCusparseStatus(cusparseDestroyMatDescr(descr_M));
        checkCusparseStatus(cusparseDestroyMatDescr(descr_L));
        checkCusparseStatus(cusparseDestroyMatDescr(descr_U));
       
        checkCusparseStatus(cusparseDestroyCsrsv2Info(info_L));
        checkCusparseStatus(cusparseDestroyCsrsv2Info(info_U));
    }

    checkCusparseStatus(cusparseDestroy(handle));

    if (settings::DEBUG) {
        fprintf(stderr, "[solver.ILU02] destroy solver completed !");
    }
}

void GPUSolverSparseILU02::validateStreamState() {
    if (settings::DEBUG_CHECK_ARG) {
        /// submitted kernel into  cuda driver
        checkCudaStatus(cudaPeekAtLastError());
        /// block and wait for execution
        checkCudaStatus(cudaStreamSynchronize(stream));
    }
}

#undef CSR

} // namespace solver