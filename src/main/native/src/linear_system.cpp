#include "linear_system.h"

//#define NF_DEBUF

/*
  cusolverStatus_t CUSOLVERAPI cusolverDnLoggerSetFile(FILE *file);
  cusolverStatus_t CUSOLVERAPI cusolverDnLoggerOpenFile(const char *logFile);
  cusolverStatus_t CUSOLVERAPI cusolverDnLoggerSetLevel(int level);
  cusolverStatus_t CUSOLVERAPI cusolverDnLoggerSetMask(int mask);
  cusolverStatus_t CUSOLVERAPI cusolverDnLoggerForceDisable();
*/

namespace errors
{

void _checkCudaStatus(cudaError_t status, size_t __line__, const char *__file__)
{
    if (status != cudaSuccess)
    {
        const char *errorName = cudaGetErrorName(status);
        const char *errorStr = cudaGetErrorString(status);
        printf("[ cuda / error ] %s#%d : cuda API failed (%d),  %s  : %s \n", __file__, (int)__line__, status,
               errorName, errorStr);
        throw std::logic_error("cuda API error");
    }
}

void _checkCuSolverStatus(cusolverStatus_t status, size_t __line__, const char *__file__)
{
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        printf("[ CuSolver / error ] %s#%d : CuSolver API failed with status %d \n", __file__, (int)__line__, status);
        throw std::logic_error("CuSolver error");
    }
}

void _checkCublasStatus(cublasStatus_t status, size_t __line__, const char *__file__)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        const char *statusName = cublasGetStatusName(status);
        const char *statusString = cublasGetStatusString(status);
        printf("[ cuBLAS / error ] %s#%d : cuBLAS API failed with status (%d) , %s : %s \n", __file__, (int)__line__,
               status, statusName, statusString);
        throw std::logic_error("cuBLAS error");
    }
}

} // namespace errors

/// cuda variables
static cusolverDnHandle_t handle = NULL;
static cublasHandle_t cublasHandle = NULL;
static cudaError_t lastError;

static cudaEvent_t start;
static cudaEvent_t end;
static float ms;

/// cusolver context
static int Lwork = 0;
static double *Workspace;

/// lu factorization output vector pivot indecies
static int *devIpiv = nullptr;

/// state from Factorization or Solver
static int *devInfo =
    nullptr; /**przyczyna INT na urzadzenieu __device__ __host__ ! ERROR 700       cudaErrorIllegalAddress */

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

    if (start != NULL)
    {
        checkCudaStatus(cudaEventDestroy(start));
    }
    if (end != NULL)
    {
        checkCudaStatus(cudaEventDestroy(end));
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

#undef CS_DEBUG

CU_SOLVER void linear_system_method_cuSolver(double *A, double *b, size_t N, cudaStream_t stream)
{
    //
    //
    //     LU Solver -  !!    this solver REQUIRMENTS -  "  n×n matrix "
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

    /// # cuBlas context
    checkCublasStatus(cublasCreate(&cublasHandle));
    checkCublasStatus(cublasSetStream(cublasHandle, stream));

    ///  # cuSolver setup solver -- nie zaincjalozowac wyrzej
    checkCuSolverStatus(cusolverDnCreate(&handle));
    checkCuSolverStatus(cusolverDnSetStream(handle, stream));

    checkCudaStatus(cudaEventCreate(&start));
    checkCudaStatus(cudaEventCreate(&end));

    ///
    /// LU - calculate the size of work buffers needed.
    ///
    ///
    ///  !!!!Remark: getrf uses fastest implementation with large workspace of size m*n
    ///
    checkCuSolverStatus(cusolverDnDgetrf_bufferSize(handle, N, N, A, N, &Lwork));

    /*




    */

    checkCudaStatus(cudaMallocAsync((void **)&Workspace, Lwork * sizeof(double), stream));

    checkCudaStatus(cudaMallocAsync((void **)&devIpiv, N * sizeof(double), stream));
    checkCudaStatus(cudaMallocAsync((void **)&devInfo, 1 * sizeof(double), stream));

    printf("[ LU ] workspace attached, 'Lwork' (workspace size)  = %d \n", Lwork);

    checkCudaStatus(cudaStreamSynchronize(stream));

    /// a tu inicjalizujemy wektory

    checkCudaStatus(cudaEventRecord(start, stream));

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

    /// Print result vector
    checkCudaStatus(cudaEventRecord(end, stream));

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
        printf("[ LU ] operation successful ! \n");
    }

    checkCudaStatus(cudaEventElapsedTime(&ms, start, end));

    // !blocking
    printf("[t] measurment %f  \n", ms);

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