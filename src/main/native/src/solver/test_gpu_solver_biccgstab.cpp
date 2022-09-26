
#include "cuda_runtime_api.h"

#include "cublas_v2.h"
#include "curand.h"
#include "cusparse.h"

#include "../cuerror.h"

#include "../solver/gpu_solver_dense_lu.h"
#include "../solver/gpu_solver_sparse_bicgstab.h"

#include "../settings.h"

#include "../quda.h"
#include "../mmio/mmio.h"

template <typename TYPE>
int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, int *m, int *n, int *nnz, TYPE **aVal, int **aRowInd, int **aColInd, int extendSymMatrix,
                       bool structuralZero);

#define CMD_FUNCTION(result) result
#define DEVICE_ZERO 0

#define HANDLE_STATUS status =

namespace test {

#ifndef IDX2C
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#endif
#ifndef IDX2R
#define IDX2R(i, j, ld) (((i) * (ld)) + (j))
#endif

void RandomMatrixBandwith(int n, double C[], int bandwidth) {
    srand(1300491);
    for (int i = 0; i < n; i++) {
        double first = ((double)(rand() % 100) + 1) / 100.0;

        C[IDX2C(i, i, n)] = 2.0 + first;

        for (int j = 1; j < bandwidth; j++) {
            if ((i + j) < n) {
                double first = ((double)(rand() % 100) + 1) / 100.0;
                double second = ((double)(rand() % 100) + 1) / 100.0;
                // C[IDX2C(i, (i + j), n)] = first;
                C[IDX2C((i + j), i, n)] = second;
            }
        }
    }
}

/// <summary>
/// host SpGeMV - dense matrix * vector
/// </summary>
/// <param name="n"></param>
/// <param name="tensorA"></param>
/// <param name="vecX"></param>
/// <param name="vecY"></param>
void DnGemv(int n, double tensorA[], double vecX[], double vecY[]) {
    /// host generalize matrix vector multiply
    for (int k = 0; k < n; k++) {
        double vdot = 0.0;
        for (int q = 0; q < n; q++) {
            vdot += tensorA[IDX2C(k, q, n)] * vecX[q]; /// IDX2C - column order , IDX2R row order
        }
        vecY[k] = vdot;
    }
}

/// <summary>
/// stdout vector
/// </summary>
/// <param name="w"></param>
/// <param name="vector"></param>
void writeOut(int w, double vector[]) {
    for (int q = 0; q < w; q++) {
        fprintf(stdout, " , %f", vector[q]);
    }
    fprintf(stdout, "\n");
}

/// <summary>
/// stdout tensor
/// </summary>
/// <param name="w"></param>
/// <param name="h"></param>
/// <param name="tensor"></param>
void writeOut(int w, int h, double tensor[]) {
    for (int i = 0; i < h; i++) {
        for (int q = 0; q < w; q++) {
            fprintf(stdout, ", %-8.5f", tensor[IDX2C(i, q, w)]);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}

#undef IDX2C
#undef IDX2R

} // namespace test

void convertToCsr(int m, int *nnz, double *d_A, int *d_csrRowPtrA, int **d_csrColIndA, double **d_csrValA) {

    cudaError_t status;
    cusparseStatus_t cusparseStatus;
    cusparseHandle_t handle;
    checkCusparseStatus(cusparseCreate(&handle));

    cusparseDnMatDescr_t matA;
    cusparseSpMatDescr_t matB;
    cusparseDenseToSparseAlg_t alg = CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;
    size_t bufferSize;
    void *buffer;

    /// --- matA
    checkCusparseStatus(cusparseCreateDnMat(&matA, m, m, m, d_A, CUDA_R_64F, CUSPARSE_ORDER_COL));

    /// --- matB
    checkCusparseStatus(
        cusparseCreateCsr(&matB, m, m, 0, d_csrRowPtrA, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    /// ---------------------------------------------

    /// --- bufferSize
    checkCusparseStatus(cusparseDenseToSparse_bufferSize(handle, matA, matB, alg, &bufferSize));
    checkCudaStatus(cudaMalloc((void **)&buffer, bufferSize));

    checkCusparseStatus(cusparseDenseToSparse_analysis(handle, matA, matB, alg, buffer));

    int64_t realNnz;
    int64_t rows;
    int64_t cols;
    checkCusparseStatus(cusparseSpMatGetSize(matB, &rows, &cols, &realNnz));
    *nnz = realNnz;

    checkCudaStatus(cudaMalloc((void **)d_csrColIndA, realNnz * sizeof(int)));
    checkCudaStatus(cudaMalloc((void **)d_csrValA, realNnz * sizeof(double)));

    checkCusparseStatus(cusparseCsrSetPointers(matB, d_csrRowPtrA, *d_csrColIndA, *d_csrValA));

    /// ---------------------------------------------
    checkCusparseStatus(cusparseDenseToSparse_convert(handle, matA, matB, alg, buffer));

    checkCusparseStatus(cusparseDestroySpMat(matB));
    checkCusparseStatus(cusparseDestroyDnMat(matA));
    checkCusparseStatus(cusparseDestroy(handle));
    checkCudaStatus(cudaFree(buffer));
}

void inspectCSR(int m, int nnz, int *d_csrRowPtrA, int *d_csrColIndA, double *d_csrValA) {
    utility::host_vector<int> h_csrRowPtrA = utility::dev_ptr<int>(d_csrRowPtrA, m + 1);
    utility::host_vector<int> h_csrColIndA = utility::dev_ptr<int>(d_csrColIndA, nnz);
    utility::host_vector<double> h_csrValA = utility::dev_ptr<double>(d_csrValA, nnz);

    for (int i = 0; i < m + 1; ++i) {
        fprintf(stdout, "csr row ptr %d ( %d ) \n", i, h_csrRowPtrA[i]);
    }
    printf("\n");
    for (int i = 0; i < nnz; ++i) {
        fprintf(stdout, "val ( %d  %d ) = %e \n", i, h_csrColIndA[i], h_csrValA[i]);
    }
    printf("\n");
}

void denseLUIsSingular(int ld, int n, double *A, double *b, int *singularity, cudaStream_t stream) {

    double *d_A;
    double *d_b;

    solver::GPUSolverDenseLU solver(stream);

    checkCudaStatus(cudaMalloc((void **)&d_A, n * n * sizeof(double)));
    checkCudaStatus(cudaMalloc((void **)&d_b, n * sizeof(double)));
    checkCudaStatus(cudaMemcpy(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice));

    settings::DEBUG_CHECK_ARG.reset(true);
    settings::DEBUG.reset(true);

    solver.solveSystem(d_A, d_b, n);

    double *a_b;
    checkCudaStatus(cudaMallocHost((void **)&a_b, n * sizeof(double)));
    checkCudaStatus(cudaMemcpy(a_b, d_b, n * sizeof(double), cudaMemcpyHostToDevice));

    printf("\n LU  result `x ( a_b )= \n");
    // test::writeOut(n, a_b);

    checkCudaStatus(cudaFree(d_A));
    checkCudaStatus(cudaFree(d_b));
    checkCudaStatus(cudaFreeHost(a_b));
}

/// <summary>
/// BiCGSTAB iterative solver for non-symetric general matricies !
/// </summary>
/// <param name="argc"></param>
/// <param name="argv"></param>
/// <returns></returns>
CMD_FUNCTION(int) main_(int argc, char *argv[]) {

    cudaError_t status;
    cudaStream_t stream;

    checkCudaStatus(cudaSetDevice(DEVICE_ZERO));
    checkCudaStatus(cudaStreamCreate(&stream));
    /// ---------------------------------------------------------

    solver::GPUSolverSparseBiCGSTAB solver(stream);
    /// ---------------------------------------------------------
    /// test real matricies 1024 * 1024 !
    const int W = 32 * 2 * 4;
    const int m = W * 16, n = W * 16;
    int nnz = m * n;

    double *d_Ar = NULL; // m *n
    double *h_Ar = NULL; // m *n
    double *h_A = NULL;  // m *n
    double h_x[n] = {};
    double h_rx[n] = {};
    double h_b[n] = {};

    double *d_A = NULL;
    double *d_b = NULL;
    double *d_x = NULL;

    checkCudaStatus(cudaMallocHost((void **)&h_Ar, m * n * sizeof(double)));
    checkCudaStatus(cudaMallocHost((void **)&h_A, m * n * sizeof(double)));

    /// macierz blokowa  `A utils::Dif2Matrix(n, h_A);

    for (int i = 0; i < m; i++)
        h_b[i] = 1.0;
    for (int i = 0; i < m; i++)
        h_x[i] = 1.0;
    // -(i % 4) / 4.0;

    /// =============
    test::RandomMatrixBandwith(m, h_A, 4);

    fprintf(stdout, "\n h_Ar = \n");
    test::writeOut(m, m, h_A);
    // utils::writeOutDiagonal(m, h_A);

    /// b = A * x
    test::DnGemv(m, h_A, h_x, h_b);

    int singularity_lu = 0;

    denseLUIsSingular(m, m, h_A, h_b, &singularity_lu, stream);

    printf("\n x = \n");
    // test::writeOut(m, h_x);
    printf("\n przed == > b = \n");
    // test::writeOut(m, h_b);

    printf("\n h_A = \n");
    // test::writeOut(m, n, h_A);

    for (int i = 0; i < m; i++)
        h_x[i] = 1.0 / (1 + 10 * (i + 1)); /// warunek poczatkowy decyduje o zbierznosci albo nie zbierznosci ! -minus
    /// bliskie zera 0.0001

    /// ---------------------------------------------------------
    int *d_csrRowPtrA = NULL;
    int *d_csrColIndA = NULL;
    double *d_csrValA = NULL;

    checkCudaStatus(cudaMalloc((void **)&d_csrRowPtrA, (m + 1) * sizeof(int)));
    checkCudaStatus(cudaMalloc((void **)&d_x, n * sizeof(double)));
    checkCudaStatus(cudaMalloc((void **)&d_b, n * sizeof(double)));

    checkCudaStatus(cudaMalloc((void **)&d_A, n * n * sizeof(double)));

    checkCudaStatus(cudaMemcpy(d_A, h_A, n * n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy(d_x, h_x, n * sizeof(double), cudaMemcpyHostToDevice));

    int singularity;
    /// ---------------------------------------------------------
    ///  DENSE A  to CSR format
    convertToCsr(m, &nnz, d_A, d_csrRowPtrA, &d_csrColIndA, &d_csrValA);

    /// inspect CSR \A
    // inspectCSR(m, nnz, d_csrRowPtrA, d_csrColIndA, d_csrValA);

    /// https://www.mathworks.com/help/matlab/ref/bicgstab.html
    ///
    /// configuracja ILU_02 dlaczego ??
    ///
    /// Since A is nonsymmetric, use ilu to generate the preconditioner M=L U.
    /// Specify a drop tolerance to ignore nondiagonal entries with values smaller than 1e-6.
    ///
    /// ---------------------------------------------------------
    solver.solveSystem(m, n, nnz, d_csrRowPtrA, d_csrColIndA, d_csrValA, d_b, d_x, &singularity);

    /// ---------------------------------------------------------
    checkCudaStatus(cudaMemcpy(h_rx, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    printf("\n result h_x = \n");
    test::writeOut(10, h_rx);

    // b = A * x
    test::DnGemv(m, h_A, h_rx, h_b);
    printf("\n h_b = \n");
    test::writeOut(10, h_b);

    checkCudaStatus(cudaFree(d_csrRowPtrA));
    checkCudaStatus(cudaFree(d_csrColIndA));
    checkCudaStatus(cudaFree(d_csrValA));
    checkCudaStatus(cudaFree(d_x));
    checkCudaStatus(cudaFree(d_b));

    return 0;
}

void cusparseSpMv_routine(int m, int n, int nnz, int *csrRowPtrA, int *csrColIndA, double *csrValA, double *inX, double *outY, cudaStream_t stream) {
    cusparseHandle_t handle;
    cusparseStatus_t status;
    ///
    /// cusparse handle
    HANDLE_STATUS cusparseCreate(&handle);
    if (status) {
        fprintf(stderr, "[solver.error] handler failure;  %s  \n", cusparseGetErrorName(status));
        exit(-1);
    }
    HANDLE_STATUS cusparseSetStream(handle, stream);
    if (status) {
        fprintf(stderr, "[cusolver/error] set stream failure;  %s  \n", cusparseGetErrorName(status));
        exit(-1);
    }
    /// -----
    cusparseSpMatDescr_t matA = NULL;
    /// input vector from SpMV
    cusparseDnVecDescr_t vecX = NULL;
    /// output vector from SpMV
    cusparseDnVecDescr_t vecY = NULL;
    cudaDataType computeType = CUDA_R_64F;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;

    /// CUSPARSE_SPMV_CSR_ALG1 = 2, CUSPARSE_SPMV_CSR_ALG2 = 3,
    cusparseSpMVAlg_t alg = CUSPARSE_SPMV_CSR_ALG2;
    cusparseIndexType_t indxeType = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;

    /// `static tensor A - SparseMV
    checkCusparseStatus(cusparseCreateCsr(&matA, m, n, nnz, csrRowPtrA, csrColIndA, csrValA, indxeType, indxeType, idxBase, computeType));
    checkCusparseStatus(cusparseCreateDnVec(&vecX, m, inX, CUDA_R_64F));
    checkCusparseStatus(cusparseCreateDnVec(&vecY, m, outY, CUDA_R_64F));

    void *externalBuffer;
    size_t bufferSize;
    double alpha = 1.0;
    double beta = 0.0;
    checkCusparseStatus(cusparseSpMV_bufferSize(handle, opA, &alpha, matA, vecX, &beta, vecY, computeType, alg, &bufferSize));

    /// routine buffer
    checkCudaStatus(cudaMallocAsync((void **)&externalBuffer, bufferSize, stream));

    /// instruction : SpMV
    checkCusparseStatus(cusparseSpMV(handle, opA, &alpha, matA, vecX, &beta, vecY, computeType, alg, externalBuffer));

    /// close context
    checkCudaStatus(cudaStreamSynchronize(stream));
    checkCudaStatus(cudaFree(externalBuffer));
    HANDLE_STATUS cusparseDestroy(handle);
    if (status) {
        fprintf(stderr, "[cusolver/error] destroy cusparse handle;  %s  \n", cusparseGetErrorName(status));
        exit(-1);
    }
}

void stdoutTensorCsr(int m, int n, int nnz, double *aVal, int *aRowInd, int *aColInd, const char *title) {
    fprintf(stdout, "tensor : %s \n", title);
    for (int j = 0; j < m; j++) {
        int from = aRowInd[j];
        int to = aRowInd[j + 1];
        fprintf(stdout, " wiersz := %d \n", j);
        for (int i = from; i < to; i++) {
            fprintf(stdout, " %e  , ", aVal[i]);
        }
        fprintf(stdout, "\n");
    }

    fprintf(stdout, "------------\n");
}

/// <summary>
/// read mtx tensor
/// </summary>
/// <param name="argc"></param>
/// <param name="argv"></param>
/// <returns></returns>
CMD_FUNCTION(int) main(int argc, char *argv[]) {
    int err;
    // ILU boost - UNSYMETRIC
     char *filename = "data\\fidap027.mtx";/// real non-symetric general
    // char *filename = "data\\fidap031.mtx";/// real non-symetric general
    //char *filename = "data\\fidap037.mtx"; /// real non-symetric general
    // char *filename = "data\\fidap021.mtx"; /// real Unsymmetri
    // char *filename = "data\\fidap026.mtx"; /// real unsymmetric
    // char *filename = "data\\s3dkt3m2.mtx"; /// real non-symetric general - diagonal strong

    //  SYMETRIC
    // char *filename = "data\\494_bus.mtx"; /// real Symmetric indefinite symmetric positive definite
    // char *filename = "data\\bfw782b.mtx"; /// real Symmetric indefinite
    // char *filename = "data\\s3rmt3m3.mtx"; /// real Symmetric positive defined
    // char *filename = "data\\s3rmq4m1.mtx"; /// real Symmetric positive definite
    // char *filename = "data\\nos5.mtx";      /// real Symmetric positive definite

    char elem_type = 'R';
    bool csrFormat = true; /// or CSC (false)
    int m = 0;
    int n = 0;
    int nnz = 0;
    double *aVal = NULL;
    int *aRowInd = NULL;
    int *aColInd = NULL;
    int extendSymMatrix = 0;
    bool structuraZero = true;
    err = loadMMSparseMatrix<double>(filename, elem_type, csrFormat, &m, &n, &nnz, &aVal, &aRowInd, &aColInd, extendSymMatrix, structuraZero);

    if (err) {
        fprintf(stderr, "errro code %d \n", err);
        exit(-1);
    }

    fprintf(stdout, "wypelnienie %d x %d  nnz %d  - wypelnienie %5.3f \n", m, n, nnz, ((double)nnz) / (m * n));

#define DEFAULT_DEVICE 0

    checkCudaStatus(cudaSetDevice(DEFAULT_DEVICE));

#undef DEFAULT_DEVICE
    cudaStream_t stream;
    checkCudaStatus(cudaStreamCreate(&stream));

    int *d_csrRowPtr = NULL;
    int *d_csrColInd = NULL;
    double *d_csrVal = NULL;

    double *d_vecX = NULL;
    double *d_vecY = NULL; /// right hand side

    /// host allocations
    double *vecY = NULL;
    double *vecX = NULL;
    checkCudaStatus(cudaMallocHost((void **)&vecY, m * sizeof(double)));
    checkCudaStatus(cudaMallocHost((void **)&vecX, m * sizeof(double)));

    double scale = 1.0;

    for (int i = 0; i < m; ++i) {
        vecX[i] = 1 * scale;
    }
    /// device allocations
    checkCudaStatus(cudaMallocAsync((void **)&d_csrRowPtr, (m + 1) * sizeof(int), stream));
    checkCudaStatus(cudaMallocAsync((void **)&d_csrColInd, nnz * sizeof(int), stream));
    checkCudaStatus(cudaMallocAsync((void **)&d_csrVal, nnz * sizeof(double), stream));
    checkCudaStatus(cudaMallocAsync((void **)&d_vecX, m * sizeof(double), stream));
    checkCudaStatus(cudaMallocAsync((void **)&d_vecY, m * sizeof(double), stream));
    /// visible on device
    checkCudaStatus(cudaMemcpyAsync(d_csrRowPtr, aRowInd, (m + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(d_csrColInd, aColInd, nnz * sizeof(int), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(d_csrVal, aVal, nnz * sizeof(double), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(d_vecX, vecX, m * sizeof(double), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaStreamSynchronize(stream));

    cusparseSpMv_routine(m, n, nnz, d_csrRowPtr, d_csrColInd, d_csrVal, d_vecX, d_vecY, stream);
    checkCudaStatus(cudaStreamSynchronize(stream));

    /// stdout y
    ///
    checkCudaStatus(cudaMemcpy(vecY, d_vecY, m * sizeof(double), cudaMemcpyDeviceToHost));
    // test::writeOut(m, vecY);
    /// reset x0

    srand(33772);
    for (int i = 0; i < m; ++i) {
        vecX[i] = scale + ((rand() % 100) / 50);
    }
    checkCudaStatus(cudaMemsetAsync(d_vecX, 0, m * sizeof(double), stream));
    checkCudaStatus(cudaMemcpyAsync(d_vecX, vecX, m * sizeof(double), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaStreamSynchronize(stream));
    /// solver
    solver::GPUSolverSparseBiCGSTAB solver(stream);

    int singularity = 0;
    solver.solveSystem(m, n, nnz, d_csrRowPtr, d_csrColInd, d_csrVal, d_vecY, d_vecX, &singularity);

    checkCudaStatus(cudaMemcpyAsync(vecX, d_vecX, m * sizeof(double), cudaMemcpyDeviceToHost, stream));
    checkCudaStatus(cudaStreamSynchronize(stream));

    for (int i = 0; i < 10; ++i) {
        fprintf(stdout, "( %d ) - %e \n", i, vecX[i]);
    }

    fprintf(stdout, "\n");

    free(aVal);
    free(aRowInd);
    free(aColInd);

    return 0;
}
