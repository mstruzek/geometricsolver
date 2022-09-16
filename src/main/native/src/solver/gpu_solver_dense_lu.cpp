#include "gpu_solver_dense_lu.h"

#include "cusolverDn.h"
#include "../cuerror.h"
#include "../settings.h"

#include "stdio.h"

namespace solver {

GPUSolverDenseLU::GPUSolverDenseLU(cudaStream_t stream) : stream(stream), Workspace(NULL), devIpiv(NULL), handle(NULL) 
{
    cudaError_t lastError = cudaPeekAtLastError();
    if (lastError != cudaSuccess) {
        printf("[cuSolver]: error solver is not initialized, [ %s ] %s \n", cudaGetErrorName(lastError), cudaGetErrorString(lastError));
        exit(-1);
    }
    ///  # cuSolver initialize solver -- nie zaincjalozowac wyrzej
    checkCuSolverStatus(cusolverDnCreate(&handle));
    checkCuSolverStatus(cusolverDnSetStream(handle, stream));

    checkCudaStatus(cudaMallocAsync((void **)&devInfo, 1 * sizeof(int), stream));
}

void GPUSolverDenseLU::solveSystem(double *A, double *b, int N) {

    // ! blocking - look back
    cudaError_t lastError = cudaPeekAtLastError();
    if (lastError != cudaSuccess) {
        printf("[cuSolver]: stream with given error , [ %s ] %s \n", cudaGetErrorName(lastError), cudaGetErrorString(lastError));
        exit(-1);
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
        if (Workspace != NULL) {
            checkCudaStatus(cudaFreeAsync(Workspace, stream));
        }
        if (devIpiv) {
            checkCudaStatus(cudaFreeAsync(devIpiv, stream));
        }
        ///  !!!!Remark: getrf uses fastest implementation with large workspace of n m*n

        /// prealocate additional buffer before LU factorization
        Lwork = (int)(Lwork * settings::SOLVER_LWORK_FACTOR.value());

        checkCudaStatus(cudaMallocAsync((void **)&Workspace, Lwork * sizeof(double), stream));
        checkCudaStatus(cudaMallocAsync((void **)&devIpiv, N * sizeof(double), stream));

        checkCudaStatus(cudaStreamSynchronize(stream));

        if (settings::DEBUG) {
            printf("[ LU ] workspace attached, 'Lwork' (workspace size)  = %d \n", Lwork);
        }
    }

    ///
    /// LU factorization    --  P * A = L *U
    ///
    ///     P - permutation vector
    ///
    checkCuSolverStatus(cusolverDnDgetrf(handle, N, N, A, N, Workspace, devIpiv, devInfo));

    /// host infot
    int hInfo = 0;

    /// dont check matrix determinant
    if (settings::DEBUG_CHECK_ARG) {
        checkCudaStatus(cudaMemcpyAsync(&hInfo, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost, stream));
        checkCudaStatus(cudaStreamSynchronize(stream));
        if (hInfo < 0) {
            printf("[ LU ] error! wrong parameter %d (exclude handle)\n", -hInfo);
            exit(-1);
        }
        if (hInfo > 0) {
            printf("[ LU ] error! tensor A is not positively defined ,  diagonal U(i,i) = 0 ,  i ( %d ) \n", hInfo);
            exit(-1);
        }
    }

    if (settings::DEBUG) {
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
    if (settings::DEBUG_CHECK_ARG) {
        /// inspect computation requirments
        checkCudaStatus(cudaMemcpyAsync(&hInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
        checkCudaStatus(cudaStreamSynchronize(stream));
        ///
        if (hInfo != 0) {
            printf("[ LU ]! parameter is wrong (not counting handle). %d \n", -hInfo);
            exit(-1);
        }
    }

    /// suspected data vector to be presetend in vector b  // B
    if (settings::DEBUG) {
        printf("[ LU ] system solved ! \n");
    }
}

GPUSolverDenseLU::~GPUSolverDenseLU() {
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
    
    if (settings::DEBUG) {
        printf("[dense.LU] solver destroy success ! \n");
    }
}

} // namespace solver