#include "gpu_linear_system.h"

#include "cuerror.h"
#include "settings.h"

#include "kernel_traits.h"

#include "solver_kernel.cuh"

#include "gpu_utility.h"

#ifdef DEBUG_GPU
#define validateStream validateStreamState()
#else
#define validateStream 
#endif

/** przyczyna INT na urzadzenieu __device__ __host__ ! ERROR 700       cudaErrorIllegalAddress */

namespace solver {

///
/// Visualt Studio Communit ; > Debug > Wlasciwosci Debugowania > Debugowanie > Srodowisko:
///
///         CUSOLVERDN_LOG_LEVEL=5
///         CUSOLVERDN_LOG_MASK = 16
///

GPULinearSystem::GPULinearSystem(cudaStream_t stream) : stream(stream) {

    // ! blocking - look back
    lastError = cudaPeekAtLastError();
    if (lastError != cudaSuccess) {
        auto errorName = cudaGetErrorName(lastError);
        auto errorStr = cudaGetErrorString(lastError);
        printf("[cuSolver]: error solver is not initialized, [ %s ] %s \n", errorName, errorStr);
        exit(1);
    }
    /// reset previous errror

    /// # cuBlas context
    checkCublasStatus(cublasCreate(&cublasHandle));
    checkCublasStatus(cublasSetStream(cublasHandle, stream));

    ///  # cuSolver initialize solver -- nie zaincjalozowac wyrzej
    checkCuSolverStatus(cusolverDnCreate(&handle));
    checkCuSolverStatus(cusolverDnSetStream(handle, stream));

    checkCudaStatus(cudaMallocAsync((void **)&devInfo, 1 * sizeof(int), stream));

    cusparseStatus_t status;

    status = cusparseCreate(&cusparseHandle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        auto errorName = cusparseGetErrorName(status);
        auto errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[cusparse/error] cusparse handle failure; ( %s ) %s \n", errorName, errorStr);
        exit(1);
    }
    status = cusparseSetStream(cusparseHandle, stream);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        auto errorName = cusparseGetErrorName(status);
        auto errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[cusparse/error] cusparse stream set failure; ( %s ) %s \n", errorName, errorStr);
        exit(1);
    }

    cusolverStatus_t cusolverStatus;
    cusolverStatus = cusolverSpCreate(&cusolverSpHandle);
    if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "[cusolver/error] cusolverSP handler failure; ( %s ) \n",
                cusolverGetErrorName(cusolverStatus));
        exit(1);
    }    
    cusolverSpSetStream(cusolverSpHandle, stream);
    if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "[cusolver/error] cusolverSP set stream failure; ( %s ) \n", cusolverGetErrorName(cusolverStatus));
        exit(1);
    }
}

void GPULinearSystem::solveLinearEquation(double *A, double *b, size_t N) {
    //
    //     LU Solver -  !!    this solver REQUIRMENTS -  "  NxN tensor "
    //
    //     - cusolverDnDgetrf_bufferSize
    //
    //     - cusolverDnDgetrf
    //
    //     - cusolverDnDgetrs
    //
    // Considerations - zapis bezposrednio do zmiennych na urzadzeniu !
    //
    // ! blocking - look back
    lastError = cudaPeekAtLastError();
    if (lastError != cudaSuccess) {
        const char *errorName = cudaGetErrorName(lastError);
        const char *errorStr = cudaGetErrorString(lastError);
        printf("[cuSolver]: stream with given error , [ %s ] %s \n", errorName, errorStr);
        exit(1);
    }

    /// reset previous errror

    int preLwork = Lwork;

    int dN = (int)N;
    ///
    /// LU - calculate the n of work buffers needed.
    ///
    checkCuSolverStatus(cusolverDnDgetrf_bufferSize(handle, dN, dN, A, dN, &Lwork));

    checkCudaStatus(cudaPeekAtLastError());

    if (Lwork > preLwork) {
        // free mem
        checkCudaStatus(cudaFreeAsync(Workspace, stream));
        checkCudaStatus(cudaFreeAsync(devIpiv, stream));

        ///  !!!!Remark: getrf uses fastest implementation with large workspace of n m*n

        /// prealocate additional buffer before LU factorization
        Lwork = (int)(Lwork * settings::get()->CU_SOLVER_LWORK_FACTOR);

        checkCudaStatus(cudaMallocAsync((void **)&Workspace, Lwork * sizeof(double), stream));
        checkCudaStatus(cudaMallocAsync((void **)&devIpiv, dN * sizeof(double), stream));

        checkCudaStatus(cudaStreamSynchronize(stream));

        if (settings::get()->DEBUG) {
            printf("[ LU ] workspace attached, 'Lwork' (workspace size)  = %d \n", Lwork);
        }
    }

    ///
    /// LU factorization    --  P * A = L *U
    ///
    ///     P - permutation vector
    ///
    checkCuSolverStatus(cusolverDnDgetrf(handle, dN, dN, A, dN, Workspace, devIpiv, devInfo));

    /*

    */

    /// host infot
    int hInfo = 0;

    /// dont check matrix determinant
    if (settings::get()->DEBUG_CHECK_ARG) {
        checkCudaStatus(cudaMemcpyAsync(&hInfo, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost, stream));
        checkCudaStatus(cudaStreamSynchronize(stream));
        if (hInfo < 0) {
            printf("[ LU ] error! wrong parameter %d (exclude handle)\n", -hInfo);
            exit(1);
        }
        if (hInfo > 0) {
            printf("[ LU ] error! tensor A is not positively defined ,  diagonal U(i,i) = 0 ,  i ( %d ) \n", hInfo);
            exit(1);
        }
    }

    if (settings::get()->DEBUG) {
        printf("[ LU ] factorization success ! \n");
    }

    ///
    /// LU  - solves a linear system of multiple right-hand sides
    ///
    ///               solver linear equation A * X = B
    ///
    checkCuSolverStatus(cusolverDnDgetrs(handle, CUBLAS_OP_N, (int)N, 1, A, (int)N, devIpiv, b, (int)N, devInfo));

    /*


    */

    /// dont check arguments
    if (settings::get()->DEBUG_CHECK_ARG) {
        /// inspect computation requirments
        checkCudaStatus(cudaMemcpyAsync(&hInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
        checkCudaStatus(cudaStreamSynchronize(stream));
        ///
        if (hInfo != 0) {
            printf("[ LU ]! parameter is wrong (not counting handle). %d \n", -hInfo);
            exit(1);
        }
    }

    /// suspected data vector to be presetend in vector b  // B

    if (settings::get()->DEBUG) {
        printf("[ LU ] operation successful ! \n");
    }
}

void GPULinearSystem::vectorNorm(int n, double *x, double *result) {

    checkCublasStatus(cublasDnrm2(cublasHandle, n, x, 1, result));

    if (settings::get()->DEBUG_SOLVER_CONVERGENCE) {
        checkCudaStatus(cudaStreamSynchronize(stream));
        printf("[cublas.norm] constraint evalutated norm  = %e \n", *result);
    } else {
        // blad na cublas RESULT jest lokalny  zawsze
        /// checkCublasStatus(cublasDnrm2(cublasHandle, n, x, 1, result))
    }
}

void GPULinearSystem::cublasAPIDaxpy(int n, const double *alpha, const double *x, int incx, double *y, int incy) {

    checkCublasStatus(::cublasDaxpy(cublasHandle, n, alpha, x, incx, y, incy));

    if (settings::get()->DEBUG_SOLVER_CONVERGENCE) {
        checkCudaStatus(cudaStreamSynchronize(stream));
        printf("[cublas.norm] cublasDaxpy \n");
    }
}

/// <summary>
/// Execution plan form computation of A *x = b equation .
/// A is a Sparse Tensor. Sparse Tensor.
///
/// QR solver with reorder "symrcm".
/// </summary>
/// <param name="csrRowPtrA">IN</param>
/// <param name="csrColIndA">IN</param>
/// <param name="csrValInd">IN</param>
/// <param name="b">IN</param>
/// <param name="x">OUT dx</param>
void GPULinearSystem::solverLinearEquationSP(int m, int n, int nnz, int *csrRowPtrA, int *csrColIndA, double *csrValA,
                                             double *b, double *x, int *singularity) {

    cusparseStatus_t status;

    if (!descrA) {
        status = cusparseCreateMatDescr(&descrA);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "[cusparse/error] cusparse crate mat descr; ( %s ) %s \n", cusparseGetErrorName(status),
                    cusparseGetErrorString(status));
            exit(1);
        }
        status = cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "[cusparse/error] cusparse cusparse mat set operation ; ( %s ) %s \n",
                    cusparseGetErrorName(status), cusparseGetErrorString(status));
            exit(1);
        }
        status = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "[cusparse/error] cusparse cusparse mat set operation ; ( %s ) %s \n",
                    cusparseGetErrorName(status), cusparseGetErrorString(status));
            exit(1);
        }
        status = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "[cusparse/error] cusparse cusparse mat set operation ; ( %s ) %s \n",
                    cusparseGetErrorName(status), cusparseGetErrorString(status));
            exit(1);
        }
    }

    /// Tolerance to decide if singular or not .
    constexpr double tol = 10e-15;

    /// No ordering if reorder=0. Otherwise, symrcm (1), symamd (2),
    ///  or csrmetisnd (3) is used to reduce zero fill-in.
    constexpr int reorder = 0;

    cusolverStatus_t cusolverStatus;

    cusolverStatus = cusolverSpDcsrlsvqr(cusolverSpHandle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol,
                                         reorder, x, singularity);

    if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "[cusolver/error] csr QR solver operation failed ; ( %s ) \n",
                cusolverGetErrorName(cusolverStatus));
        exit(1);
    }
    validateStream;

    if (settings::get()->DEBUG_CHECK_ARG) {
        checkCudaStatus(cudaStreamSynchronize(stream));
        fprintf(stdout, "[cusolverSP] singularity   = %d \n", *singularity);
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
/// <param name="cooValues">IN/OUT sorted out (PT) </param>
/// <param name="csrRowPtrA">OUT vector - m + 1  [ csr compact form ] </param>
/// <param name="PT">IN/OUT vector[nnz], permutation from coo into csr</param>
/// <param name="initialize">if true PT is not reused in computation and csrRowPtrA is build</param>
void GPULinearSystem::transformToCsr(int m, int n, int nnz, int *cooRowInd, int *cooColInd, double *cooValues,
                                     int *csrRowInd, int *PT, bool initialize) {

    cusparseStatus_t status;

    /// first Xcoosort round
    if (initialize) {

        /// required minimum vectors lengths
        if (nnz > PT_nnz) {

            if (PT1 != NULL) {
                checkCudaStatus(cudaFreeAsync(PT2, stream));
            }
            if (PT2 != NULL) {
                checkCudaStatus(cudaFreeAsync(PT1, stream));
            }

            checkCudaStatus(cudaMallocAsync((void **)&PT1, nnz * sizeof(int), stream));
            checkCudaStatus(cudaMallocAsync((void **)&PT2, nnz * sizeof(int), stream));

            PT_nnz = nnz;
        }

        // first acc vector identity vector
        cusparseCreateIdentityPermutation(cusparseHandle, nnz, PT1);

        /// seconf acc identity vector
        cusparseCreateIdentityPermutation(cusparseHandle, nnz, PT2);

        /// required computation Buffer
        size_t lastBufferSize = pBufferSizeInBytes;

        status = cusparseXcoosort_bufferSizeExt(cusparseHandle, m, n, nnz, cooRowInd, cooColInd, &pBufferSizeInBytes);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            auto errorName = cusparseGetErrorName(status);
            auto errorStr = cusparseGetErrorString(status);
            fprintf(stderr, "[cusparse/error]  Xcoosort buffer size estimate failes; ( %s ) %s \n", errorName,
                    errorStr);
            exit(1);
        }

        /// async prior action with host reference
        checkCudaStatus(cudaStreamSynchronize(stream));

        if (pBufferSizeInBytes > lastBufferSize) {
            if (pBuffer) {
                checkCudaStatus(cudaFreeAsync(pBuffer, stream));
            }
            checkCudaStatus(cudaMallocAsync((void **)&pBuffer, pBufferSizeInBytes, stream));
        }

        checkCudaStatus(cudaStreamSynchronize(stream));

        /// recompute sort by column action on COO tensor
        status = cusparseXcoosortByColumn(cusparseHandle, m, n, nnz, cooRowInd, cooColInd, PT1, pBuffer);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            auto errorName = cusparseGetErrorName(status);
            auto errorStr = cusparseGetErrorString(status);
            fprintf(stderr, "[cusparse/error]  Xcoosort by column failure ; ( %s ) %s \n", errorName, errorStr);
            exit(1);
        }
        validateStream;

        /// recompute sort by row action on COO tensor
        status = cusparseXcoosortByRow(cusparseHandle, m, n, nnz, cooRowInd, cooColInd, PT2, pBuffer);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            auto errorName = cusparseGetErrorName(status);
            auto errorStr = cusparseGetErrorString(status);
            fprintf(stderr, "[cusparse/error]  Xcoosort by row failure ; ( %s ) %s \n", errorName, errorStr);
            exit(1);
        }
        validateStream;

        // prior action :: if the Stream Ordered Memory Allocator ???

        constexpr const unsigned OBJECTS_PER_THREAD = 1;
        constexpr const unsigned DEF_BLOCK_DIM = 1024;
        KernelTraits<OBJECTS_PER_THREAD, DEF_BLOCK_DIM> PermutationKernelTraits{(unsigned)nnz};

        /// Permutation vector compactio procedure PT[u] = PT1[PT2[u]]
        unsigned GRID_DIM = PermutationKernelTraits.GRID_DIM;
        unsigned BLOCK_DIM = PermutationKernelTraits.GRID_DIM;

        compactPermutationVector(GRID_DIM, BLOCK_DIM, stream, nnz, PT1, PT2, PT);
        validateStream;

        /// create csrRowPtrA    -  async execution
        status = cusparseXcoo2csr(cusparseHandle, cooRowInd, nnz, m, csrRowInd, CUSPARSE_INDEX_BASE_ZERO);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            auto errorName = cusparseGetErrorName(status);
            auto errorStr = cusparseGetErrorString(status);
            fprintf(stderr, "[cusparse/error]  conversion to csr storage format ; ( %s ) %s \n", errorName, errorStr);
            exit(1);
        }
        validateStream;

    } /// initialize end

    /// Inplace cooValus pivoting   -   async execution
    status = cusparseDgthr(cusparseHandle, nnz, cooValues, cooValues, PT, CUSPARSE_INDEX_BASE_ZERO);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        auto errorName = cusparseGetErrorName(status);
        auto errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[cusparse/error]  gather operation on cooValues; ( %s ) %s \n", errorName, errorStr);
        exit(1);
    }
}

void GPULinearSystem::invertPermuts(int n, int *PT, int *INV) {

    constexpr const unsigned OBJECTS_PER_THREAD = 4;
    constexpr const unsigned DEF_BLOCK_DIM = 1024;

    KernelTraits<OBJECTS_PER_THREAD, DEF_BLOCK_DIM> CompressKernelTraits{(unsigned)n};

    unsigned GRID_DIM = CompressKernelTraits.GRID_DIM;
    unsigned BLOCK_DIM = CompressKernelTraits.GRID_DIM;
    compactPermutationVector(GRID_DIM, BLOCK_DIM, stream, PT, INV, n);
    validateStream;
}

GPULinearSystem::~GPULinearSystem() {

    cusparseStatus_t sparseStatus;

    checkCudaStatus(cudaFreeAsync(Workspace, stream));
    checkCudaStatus(cudaFreeAsync(devIpiv, stream));
    checkCudaStatus(cudaFreeAsync(devInfo, stream));

    if (pBuffer) {
        checkCudaStatus(cudaFreeAsync(pBuffer, stream));
    }
    if (PT1) {
        checkCudaStatus(cudaFreeAsync(PT1, stream));
    }
    if (PT2) {
        checkCudaStatus(cudaFreeAsync(PT2, stream));
    }

    checkCuSolverStatus(cusolverDnDestroy(handle));

    if (descrA) {
        sparseStatus = cusparseDestroyMatDescr(descrA);
        if (sparseStatus != CUSPARSE_STATUS_SUCCESS) {
            const char *errorName = cusparseGetErrorName(sparseStatus);
            const char *errorStr = cusparseGetErrorString(sparseStatus);
            fprintf(stderr, "[cusparse] cusparse handle failure; ( %s ) %s \n", errorName, errorStr);
        }
    }

    cusolverStatus_t cusolverStatus;
    cusolverStatus = cusolverSpDestroy(cusolverSpHandle);
    if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "[cusolver/error] sovlerSP destroy; ( %s ) \n", cusolverGetErrorName(cusolverStatus));
    }

    sparseStatus = cusparseDestroy(cusparseHandle);
    if (sparseStatus != CUSPARSE_STATUS_SUCCESS) {
        const char *errorName = cusparseGetErrorName(sparseStatus);
        const char *errorStr = cusparseGetErrorString(sparseStatus);
        fprintf(stderr, "[cusparse] cusparse handle failure; ( %s ) %s \n", errorName, errorStr);
    }

    checkCublasStatus(cublasDestroy(cublasHandle));
}

void GPULinearSystem::validateStreamState() {
    if (settings::get()->DEBUG_CHECK_ARG) {
        /// submitted kernel into  cuda driver 
        checkCudaStatus(cudaPeekAtLastError()); 
        /// block and wait for execution
        checkCudaStatus(cudaStreamSynchronize(stream));        
    }
}

} // namespace solver

#undef validateStream 


