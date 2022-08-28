#include "linear_system.h"

/// cuBlas
#include "cublas_v2.h"

/// Cuda Runtime API
#include "cuda_runtime_api.h"

/// cuSolverDN - Cuda Solver Danse - infrastructure
#include "cusolverDn.h"


#include "cuerror.h"
#include "settings.h"

/// cuda variables
static cusolverDnHandle_t handle = NULL;
static cublasHandle_t cublasHandle = NULL;
static cudaError_t lastError;

/// cusolver context
static int Lwork = 0;
static double *Workspace;

/// lu factorization output vector pivot indecies
static int *devIpiv = nullptr;

/// data from Factorization or Solver
static int *devInfo = nullptr; 

/** przyczyna INT na urzadzenieu __device__ __host__ ! ERROR 700       cudaErrorIllegalAddress */


/// host infot
static int hInfo;

/// marker
#define CU_SOLVER

CU_SOLVER void linear_system_method_cuSolver_reset(cudaStream_t stream)
{
    /// mem release
    if (Workspace != nullptr)
    {
        checkCudaStatus(cudaFreeAsync(Workspace, stream));
        checkCudaStatus(cudaFreeAsync(devInfo, stream));
        checkCudaStatus(cudaFreeAsync(devIpiv, stream));
        Workspace = nullptr;
        devInfo = nullptr;
        devIpiv = nullptr;

    }

    if (handle != nullptr)
    {
        checkCuSolverStatus(cusolverDnDestroy(handle));

        handle = nullptr;
    }

    if (cublasHandle != nullptr)
    {
        checkCublasStatus(cublasDestroy(cublasHandle));
        cublasHandle = nullptr;
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
    //     LU Solver -  !!    this solver REQUIRMENTS -  "  NxN tensor "
    //
    //     = cusolverDnDgetrf_bufferSize
    //
    //     = cusolverDnDgetrf
    //
    //     = cusolverDnDgetrs
    //
    //
    // Considerations - zapis bezposrednio do zmiennych na urzadzeniu !

    // ! blocking - look back
    lastError = cudaPeekAtLastError();
    if (lastError != cudaSuccess)
    {
        const char *errorName = cudaGetErrorName(lastError);
        const char *errorStr = cudaGetErrorString(lastError);
        printf("[cuSolver]: error solver is not initialized, [ %s ] %s \n", errorName, errorStr);
        exit(1);
    }

    /// reset previous errror
    hInfo = 0;


    if (cublasHandle == nullptr)
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

    if (Lwork > preLwork || Workspace == nullptr)
    {
        // free mem
        checkCudaStatus(cudaFreeAsync(Workspace, stream));
        checkCudaStatus(cudaFreeAsync(devIpiv, stream));
        checkCudaStatus(cudaFreeAsync(devInfo, stream));

        /// prealocate additional buffer before LU factorization 
        Lwork = (int) (Lwork * settings::get()->CU_SOLVER_LWORK_FACTOR);

        checkCudaStatus(cudaMallocAsync((void **)&Workspace, Lwork * sizeof(double), stream));
        checkCudaStatus(cudaMallocAsync((void **)&devIpiv, N * sizeof(double), stream));
        checkCudaStatus(cudaMallocAsync((void **)&devInfo, 1 * sizeof(double), stream));

        if (settings::get()->DEBUG)
        {
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
    
    checkCudaStatus(cudaMemcpyAsync(&hInfo, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost, stream));


    /// dont check matrix determinant
    if (settings::get()->DEBUG_CHECK_ARG)
    {
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
            printf("[ LU ] error! tensor A is not positively defined ,  diagonal U(i,i) = 0 ,  i ( %d ) \n", hInfo);
            exit(1);
        }
    }


    if(settings::get()->DEBUG){
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


    /// inspect computation requirments
    checkCudaStatus(cudaMemcpyAsync(&hInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));


    /// dont check arguments
    if (settings::get()->DEBUG_CHECK_ARG)
    {
        checkCudaStatus(cudaStreamSynchronize(stream));

        ///
        if (hInfo != 0)
        {
            printf("[ LU ]! parameter is wrong (not counting handle). %d \n", -hInfo);
            exit(1);
        }
    }

    /// 
    /// suspected data vector to be presetend in vector b  // B

    if (settings::get()->DEBUG)
    {
        printf("[ LU ] operation successful ! \n");

    }

}

void linear_system_method_cuBlas_vectorNorm(int n, double *x, double *result, cudaStream_t stream)
{    
    double local = 0.0;

    if (settings::get()->DEBUG_SOLVER_CONVERGENCE)
    {     
        checkCublasStatus(cublasDnrm2(cublasHandle, n, x, 1, &local));        
        checkCudaStatus(cudaStreamSynchronize(stream));
        checkCudaStatus(cudaMemcpyAsync(result, &local, sizeof(double), cudaMemcpyHostToDevice));

        printf("[cublas.norm] constraint evalutated norm  = %e \n", local);
    }
    else
    {
        /// result MUST be device vector
        //cublasDnrm2(cublasHandle, n, x, 1, result);

        checkCublasStatus(cublasDnrm2(cublasHandle, n, x, 1, &local));
        checkCudaStatus(cudaStreamSynchronize(stream));
        checkCudaStatus(cudaMemcpyAsync(result, &local, sizeof(double), cudaMemcpyHostToDevice));
    } 
}