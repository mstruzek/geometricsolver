#include "gpu_linear_system.h"

#include "cuerror.h"
#include "settings.h"

#include "kernel_traits.h"

#include "solver_kernel.cuh"

#include "gpu_utility.h"
#include "quda.cuh"

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

GPULinearSystem::GPULinearSystem(cudaStream_t stream) : stream(stream), Workspace(NULL), devIpiv(NULL), handle(NULL) {

    // ! blocking - look back
    lastError = cudaPeekAtLastError();
    if (lastError != cudaSuccess) {
        auto errorName = cudaGetErrorName(lastError);
        auto errorStr = cudaGetErrorString(lastError);
        printf("[cuSolver]: error solver is not initialized, [ %s ] %s \n", errorName, errorStr);
        exit(1);
    }

    ///  # cuSolver initialize solver -- nie zaincjalozowac wyrzej
    checkCuSolverStatus(cusolverDnCreate(&handle));
    checkCuSolverStatus(cusolverDnSetStream(handle, stream));

    checkCudaStatus(cudaMallocAsync((void **)&devInfo, 1 * sizeof(int), stream));

    cusolverStatus_t status;
    status = cusolverSpCreate(&cusolverSpHandle);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "[cusolver/error] cusolverSP handler failure; ( %s ) \n", cusolverGetErrorName(status));
        exit(1);
    }
    cusolverSpSetStream(cusolverSpHandle, stream);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "[cusolver/error] cusolverSP set stream failure; ( %s ) \n",
                cusolverGetErrorName(status));
        exit(1);
    }
}

void GPULinearSystem::solveLinearEquation(double *A, double *b, int N) {
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

    ///
    /// LU - calculate the n of work buffers needed.
    ///
    checkCuSolverStatus(cusolverDnDgetrf_bufferSize(handle, N, N, A, N, &Lwork));

    checkCudaStatus(cudaPeekAtLastError());

    if (Lwork > preLwork) {
        // free mem
        if (Workspace!=NULL) {
            checkCudaStatus(cudaFreeAsync(Workspace, stream));
        }         
        if (devIpiv) {
            checkCudaStatus(cudaFreeAsync(devIpiv, stream));
        }
       
        ///  !!!!Remark: getrf uses fastest implementation with large workspace of n m*n

        /// prealocate additional buffer before LU factorization
        Lwork = (int)(Lwork * settings::get()->CU_SOLVER_LWORK_FACTOR);

        checkCudaStatus(cudaMallocAsync((void **)&Workspace, Lwork * sizeof(double), stream));
        checkCudaStatus(cudaMallocAsync((void **)&devIpiv, N * sizeof(double), stream));

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
    checkCuSolverStatus(cusolverDnDgetrf(handle, N, N, A, N, Workspace, devIpiv, devInfo));

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
    checkCuSolverStatus(cusolverDnDgetrs(handle, CUBLAS_OP_N, N, 1, A, N, devIpiv, b, N, devInfo));

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

    if (settings::get()->DEBUG_CSR_FORMAT) {
        checkCudaStatus(cudaStreamSynchronize(stream));
        utility::stdout_vector(utility::dev_vector<int>(csrRowPtrA, nnz, stream), "d_csrRowPtrA");
        utility::stdout_vector(utility::dev_vector<int>(csrColIndA, nnz, stream), "d_csrColIndA");
        utility::stdout_vector(utility::dev_vector<double>(csrValA, nnz, stream), "d_csrValA");           
    }

#define SOLVER_QR_TOLERANCE 1e-10

    /// Tolerance to decide if singular or not .
    constexpr double tol = SOLVER_QR_TOLERANCE;


#define SOLVER_QR_SCHEMA_SYMRCM 1
#define SOLVER_QR_SCHEMA_SYMAMD 2
#define SOLVER_QR_SCHEMA_CSRMETISND 3
/// No ordering if reorder=0. Otherwise, symrcm (1), symamd (2),
///  or csrmetisnd (3) is used to reduce zero fill-in.
#define SOLVER_QR_REORDER SOLVER_QR_SCHEMA_SYMRCM 

    constexpr int reorder = SOLVER_QR_REORDER;

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


GPULinearSystem::~GPULinearSystem() {

    cusparseStatus_t sparseStatus;

    if (Workspace) {
        checkCudaStatus(cudaFreeAsync(Workspace, stream));
    }
    if (devIpiv) {
        checkCudaStatus(cudaFreeAsync(devIpiv, stream));
    }
    if (devInfo) {
        checkCudaStatus(cudaFreeAsync(devInfo, stream));
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
