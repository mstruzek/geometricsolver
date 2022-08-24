#include "linear_system.h"

/// cuBlas
#include "cublas_v2.h"

/// Cuda Runtime API
#include "cuda_runtime_api.h"

/// cuSolverDN - Cuda Solver Danse - infrastructure
#include "cusolverDn.h"


#include "cuerror.h"


/// cuda variables
static cusolverDnHandle_t handle = NULL;
static cublasHandle_t cublasHandle = NULL;
static cudaError_t lastError;

/// cusolver context
static int Lwork = 0;
static double *Workspace;

/// lu factorization output vector pivot indecies
static int *devIpiv = nullptr;

/// state from Factorization or Solver
static int *devInfo = nullptr; 

/** przyczyna INT na urzadzenieu __device__ __host__ ! ERROR 700       cudaErrorIllegalAddress */


/// host infot
static int hInfo;

/// marker
#define CU_SOLVER

CU_SOLVER void linear_system_method_cuSolver_reset(cudaStream_t stream)
{

    // mem release
    if (Workspace != NULL)
    {
        checkCudaStatus(cudaFreeAsync(Workspace, stream));
        checkCudaStatus(cudaFreeAsync(devInfo, stream));
    }


    if (handle != NULL)
    {
        checkCuSolverStatus(cusolverDnDestroy(handle));
    }
    if (cublasHandle != NULL)
    {
        checkCublasStatus(cublasDestroy(cublasHandle));
    }
}


///
/// Visualt Studio Communit ; > Debug > Wlasciwosci Debugowania > Debugowanie > Srodowisko: 
/// 
///         CUSOLVERDN_LOG_LEVEL=5
///         CUSOLVERDN_LOG_MASK = 16
///
#define CS_DEBUG
#undef CS_DEBUG

CU_SOLVER void linear_system_method_cuSolver(double *A, double *b, size_t N, cudaStream_t stream)
{
    //
    //
    //     LU Solver -  !!    this solver REQUIRMENTS -  "  n�n matrix "
    //
    //
    //      cusolverDnDgetrf_bufferSize
    //
    //      cusolverDnDgetrf
    //
    //      cusolverDnDgetrs
    //
    //
    //
    //
    // Considerations - zapis bezposrednio do zmiennych na urzadzeniu !

    // ! blocking - look back
    lastError = cudaGetLastError();
    if (lastError != cudaSuccess)
    {
        printf("[cuSolver]: error solver is not initialized \n");
        exit(1);
    }

    /// reset previous errror
    hInfo = 0;


    if (cublasHandle == NULL)
    {
        /// # cuBlas context
        checkCublasStatus(cublasCreate(&cublasHandle));
        checkCublasStatus(cublasSetStream(cublasHandle, stream));

        ///  # cuSolver setup solver -- nie zaincjalozowac wyrzej
        checkCuSolverStatus(cusolverDnCreate(&handle));
        checkCuSolverStatus(cusolverDnSetStream(handle, stream));
    }

    int preLwork = Lwork;

    ///
    /// LU - calculate the size of work buffers needed.
    ///
    ///
    ///  !!!!Remark: getrf uses fastest implementation with large workspace of size m*n
    ///
    checkCuSolverStatus(cusolverDnDgetrf_bufferSize(handle, N, N, A, N, &Lwork));

    if (Lwork > preLwork)
    {
        // free mem
        checkCudaStatus(cudaFreeAsync(Workspace, stream));
        checkCudaStatus(cudaFreeAsync(devIpiv, stream));
        checkCudaStatus(cudaFreeAsync(devInfo, stream));

        checkCudaStatus(cudaMallocAsync((void **)&Workspace, Lwork * sizeof(double), stream));
        checkCudaStatus(cudaMallocAsync((void **)&devIpiv, N * sizeof(double), stream));
        checkCudaStatus(cudaMallocAsync((void **)&devInfo, 1 * sizeof(double), stream));

        printf("[ LU ] workspace attached, 'Lwork' (workspace size)  = %d \n", Lwork);
    }


#ifdef CS_DEBUG
    checkCudaStatus(cudaStreamSynchronize(stream));    
#endif    

    ///
    /// LU factorization    --  P * A = L *U
    ///
    ///     P - permutation vector
    ///
    checkCuSolverStatus(cusolverDnDgetrf(handle, N, N, A, N, Workspace, devIpiv, devInfo));

    /*





    */

    checkCudaStatus(cudaMemcpyAsync(&hInfo, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost, stream));

    checkCudaStatus(cudaStreamSynchronize(stream));

    // !blocking
    if (hInfo < 0)
    {
        printf("[ LU ] error! wrong parameter %d (exclude handle)\n", -hInfo);
        exit(1);
    }

    // !blocking
    if (hInfo > 0)
    {
        printf("[ LU ] error! is not positive defined ,  U(i,i) = 0 ,  i( %d ) \n", hInfo);
        exit(1);
    }

    if (hInfo == 0)
    {

#ifdef CS_DEBUG
        printf("[ LU ] factorization success ! \n");
#endif CS_DEBUG

    }

    ///
    /// LU  - solves a linear system of multiple right-hand sides
    ///
    ///               solver linear equation A * X = B
    ///
    checkCuSolverStatus(cusolverDnDgetrs(handle, CUBLAS_OP_N, N, 1, A, N, devIpiv, b, N, devInfo));

    /*





    */

    /// inspect computation requirments
    checkCudaStatus(cudaMemcpyAsync(&hInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));


    ///  b - is output vector

    checkCudaStatus(cudaStreamSynchronize(stream));

    // !blocking
    if (hInfo != 0)
    {
        printf("[ LU ]! parameter is wrong (not counting handle). %d \n", -hInfo);
        exit(1);
    }

    if (hInfo == 0)
    {

#ifdef CS_DEBUG
        printf("[ LU ] operation successful ! \n");
#endif
    }

    /// suspected state vector to be presetend in vector b
}

void linear_system_method_cuBlas_vectorNorm(int n, double *x, double *result, cudaStream_t stream)
{

    //
    // checkCublasStatus(cublasSetStream(cublasHandle, stream));

    double result_host = 0;
    // Initilize async request
    cublasDnrm2(cublasHandle, n, x, 1, &result_host);

    checkCudaStatus(cudaStreamSynchronize(stream));

    printf(" epsilon = %f \n", result_host);

    cudaMemcpy(result, &result_host, 1 * sizeof(double), cudaMemcpyHostToDevice);
}