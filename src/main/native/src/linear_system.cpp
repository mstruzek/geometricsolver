#include "linear_system.h"


namespace errors {

void _checkCudaStatus(cudaError_t status, size_t __line) {
    if (status != cudaSuccess) {
        printf("%li: cuda API failed with status %d\n", __line, status);
        throw std::logic_error("cuda API error");
    }
}

void _checkCuSolverStatus(cusolverStatus_t status, size_t _line_) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        printf("%li: CuSolver API failed with status %d\n", _line_, status);
        throw std::logic_error("CuSolver error");
    }
}

void _checkCublasStatus(cublasStatus_t status, size_t __line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("%li: cuBLAS API failed with status %d\n", __line, status);
        throw std::logic_error("cuBLAS error");
    }
}

} // namespace errors




/// cuda variables
cusolverDnHandle_t handle = NULL;
cudaStream_t stream = NULL;
cudaError_t lastError;

cudaEvent_t start;
cudaEvent_t end;
float ms;

/// cusolver context
int Lwork = 0;
void *Workspace;
int *info; /**przyczyna INT na urzadzenieu ! BLAD 700       cudaErrorIllegalAddress */
int hinfo;

/// marker
#define CU_SOLVER  


CU_SOLVER void linear_system_method_0_reset() {

    if (stream == NULL)
        return ;

/// mem release
    checkCudaStatus(cudaFree(Workspace));
    checkCudaStatus(cudaFree(info));

    checkCudaStatus(cudaEventDestroy(start));
    checkCudaStatus(cudaEventDestroy(end));

    checkCudaStatus(cudaStreamDestroy(stream));        
    checkCuSolverStatus(cusolverDnDestroy(handle));
    
}



CU_SOLVER void linear_system_method_0(double *A, double *b, size_t N) 
{
    if (stream == NULL) {

        /// setup solver
        checkCudaStatus(cudaStreamCreate(&stream));
        checkCuSolverStatus(cusolverDnCreate(&handle));
        checkCuSolverStatus(cusolverDnSetStream(handle, stream));

        lastError = cudaGetLastError();
        if (lastError != cudaSuccess) {
            printf("[cuSolver]: error solver is not initialized \n");
            exit(1);
        }
                 
        checkCudaStatus(cudaEventCreate(&start));
        checkCudaStatus(cudaEventCreate(&end));

    /// Lwork * 2 !!!
        checkCuSolverStatus(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, N, A, N, &Lwork));        

        checkCudaStatus(cudaMalloc((void **)&Workspace, Lwork * sizeof(data_type)));
        checkCudaStatus(cudaMalloc((void **)&info, sizeof(int)));
        checkCudaStatus(cudaMemset(info, 0, sizeof(int)));

        printf("workspace attached, 'Lwork' (workspace size)  = %d \n", Lwork);
    }

    /// a tu inicjalizujemy wektory

    checkCudaStatus(cudaEventRecord(start, stream));

/// Cholesky Factorization

    checkCuSolverStatus(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER, N, A, N, (data_type *)Workspace, Lwork, info));

    // checkCudaStatus(cudaStreamSynchronize(stream));

    checkCudaStatus(cudaMemcpy(&hinfo, info, sizeof(int), cudaMemcpyDeviceToHost));

    if (hinfo < 0) {
        printf("error! wrong parameter %d \n", hinfo);
        goto solver_exit;
    }

    if (hinfo > 0) {
        printf("error! leading minor is not positive definite %d \n", hinfo);
        goto solver_exit;
    }

    ///
    /// Solver Linear Equation A * X = B
    ///
    checkCuSolverStatus(cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_LOWER, N, 1, A, N, b, N, info));


/// inspect computation requirments
    checkCudaStatus(cudaMemcpy(&hinfo, info, sizeof(int), cudaMemcpyDeviceToHost));
    if (hinfo != 0) {
        printf("error! wrong parameter %d \n", hinfo);
        goto uexit;
    }

    /// Print result vector
    checkCudaStatus(cudaEventRecord(end, stream));

    /// hx = b
    
///  b - is output vector
    checkCudaStatus(cudaMemcpyAsync(hb, b, N * sizeof(data_type), cudaMemcpyDeviceToHost, stream));
    
    checkCudaStatus(cudaStreamSynchronize(stream));

    checkCudaStatus(cudaEventElapsedTime(&ms, start, end));

    printf(" [r] time measurment %f  \n", ms);

/// suspected state vector to be presetend in vector b

}