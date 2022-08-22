#include "linear_system.h"


//#define NF_DEBUF



/*
  cusolverStatus_t CUSOLVERAPI cusolverDnLoggerSetFile(FILE *file);
  cusolverStatus_t CUSOLVERAPI cusolverDnLoggerOpenFile(const char *logFile);
  cusolverStatus_t CUSOLVERAPI cusolverDnLoggerSetLevel(int level);
  cusolverStatus_t CUSOLVERAPI cusolverDnLoggerSetMask(int mask);
  cusolverStatus_t CUSOLVERAPI cusolverDnLoggerForceDisable();
*/


namespace errors {

void _checkCudaStatus(cudaError_t status, size_t __line) {
    if (status != cudaSuccess) {
        const char *errorName = cudaGetErrorName(status);
        const char *errorStr = cudaGetErrorString(status);
        printf("[ cuda / error ] L#%d : cuda API failed (%d),  %s  : %s \n", (int)__line, status, errorName, errorStr);
        throw std::logic_error("cuda API error");
    }
}

void _checkCuSolverStatus(cusolverStatus_t status, size_t _line_) {
    if (status != CUSOLVER_STATUS_SUCCESS) {       
        printf("[ CuSolver / error ] %d : CuSolver API failed with status %d \n", (int) _line_, status);
        throw std::logic_error("CuSolver error");
    }
}

void _checkCublasStatus(cublasStatus_t status, size_t __line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        const char *statusName = cublasGetStatusName(status);
        const char *statusString = cublasGetStatusString(status);
        printf("[ cuBLAS / error ] L#%d : cuBLAS API failed with status (%d) , %s : %s \n", (int) __line, status, statusName, statusString);
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
static void *Workspace;
static int *info; /**przyczyna INT na urzadzenieu __device__ __host__ ! ERROR 700       cudaErrorIllegalAddress */
static int hinfo;

/// marker
#define CU_SOLVER  

CU_SOLVER void linear_system_method_cuSolver_reset(cudaStream_t stream) {

// mem release
    if (Workspace != NULL) {
        checkCudaStatus(cudaFreeAsync(Workspace, stream));
        checkCudaStatus(cudaFreeAsync(info, stream));
    }

    if (start != NULL) {
        checkCudaStatus(cudaEventDestroy(start));
    }
    if (end != NULL) {
        checkCudaStatus(cudaEventDestroy(end));
    }          

    if (handle != NULL) {
        checkCuSolverStatus(cusolverDnDestroy(handle));
    }
    if (cublasHandle != NULL) {
        checkCublasStatus(cublasDestroy(cublasHandle));
    }          
}

#undef CS_DEBUG

CU_SOLVER void linear_system_method_cuSolver(double *A, double *b, size_t N, cudaStream_t stream) 
{   
    // Considerations - zapis bezposrednio do zmiennych na urzadzeniu !

    // ! blocking - look back 
    lastError = cudaGetLastError();
    if (lastError != cudaSuccess)
    {
        printf("[cuSolver]: error solver is not initialized \n");
        exit(1);
    }


    /// # cuBlas context
    checkCublasStatus(cublasCreate(&cublasHandle));
    checkCublasStatus(cublasSetStream(cublasHandle, stream));
       
    ///  # cuSolver setup solver -- nie zaincjalozowac wyrzej
    checkCuSolverStatus(cusolverDnCreate(&handle));        
    checkCuSolverStatus(cusolverDnSetStream(handle, stream));


    checkCudaStatus(cudaEventCreate(&start));
    checkCudaStatus(cudaEventCreate(&end));

    /// Lwork * 1.5 !!!
    checkCuSolverStatus(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, N, A, N, &Lwork));        

    checkCudaStatus(cudaMallocAsync((void **)&Workspace, Lwork * sizeof(double), stream));
    checkCudaStatus(cudaMallocAsync((void **)&info, sizeof(int), stream));
    checkCudaStatus(cudaMemsetAsync(info, 0, sizeof(int)), stream);

    // ! blocking NDEBUG
    printf("workspace attached, 'Lwork' (workspace size)  = %d \n", Lwork);
    

    checkCudaStatus(cudaStreamSynchronize(stream));    

    /// a tu inicjalizujemy wektory

    checkCudaStatus(cudaEventRecord(start, stream));

/// Cholesky Factorization -- near error < 10e-10 step preserve old factorization !

    checkCuSolverStatus(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_UPPER, N, A, N, (double *)Workspace, Lwork, info));    

    checkCudaStatus(cudaMemcpyAsync(&hinfo, info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    
    checkCudaStatus(cudaStreamSynchronize(stream));

    // !blocking 
    if (hinfo < 0) {
        printf("error! wrong parameter %d \n", hinfo);
        exit(1);
    }

    // !blocking
    if (hinfo > 0) {
        printf("error! leading minor is not positive definite %d \n", hinfo);
        exit(1);
    }

    ///
    /// Solver Linear Equation A * X = B
    ///
    checkCuSolverStatus(cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_UPPER, N, 1, A, N, b, N, info));


/// inspect computation requirments
    checkCudaStatus(cudaMemcpyAsync(&hinfo, info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    /// Print result vector
    checkCudaStatus(cudaEventRecord(end, stream));
    
///  b - is output vector
    
    checkCudaStatus(cudaStreamSynchronize(stream));

    // !blocking
    if (hinfo != 0) {
        printf("error! wrong parameter %d \n", hinfo);
        exit(1);
    }


    checkCudaStatus(cudaEventElapsedTime(&ms, start, end));

    // !blocking
    printf("[t] measurment %f  \n", ms);

/// suspected state vector to be presetend in vector b

}


void linear_system_method_cuBlas_vectorNorm(int n, double *x, double *result, cudaStream_t stream) {
    
    //
    checkCublasStatus(cublasSetStream(cublasHandle, stream));

    // Initilize async request 
    cublasDnrm2(cublasHandle, n, x, 1, result);
}