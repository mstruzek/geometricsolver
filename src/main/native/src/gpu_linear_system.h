#ifndef _LINEAR_SYSTEM_CUH_
#define _LINEAR_SYSTEM_CUH_


#include "cuda_runtime_api.h"       /// Cuda Runtime API
#include "cublas_v2.h"              /// cuBlas

#include "cusparse.h"
#include "cusolverDn.h"             /// cuSolverDN - Cuda Solver Danse - infrastructure
#include "cusolverSp.h"             /// cuSolverSP - Cuda Solver Sparse - infrastructure


namespace solver {

class GPULinearSystem {
  public:

    GPULinearSystem(cudaStream_t stream);

    /// <summary>
    /// Execution plan for computation of system of linear equations- A * x  = B . 
    /// A is a dense tensor !
    /// 
    /// LU solver 
    /// - cusolverDnDgetrf_bufferSize - evaluate neede computation buffer 
    /// - cusolverDnDgetrf  - LU factorization routine 
    /// - cusolverDnDgetrs  - LU solver routine
    /// </summary>
    /// <param name="A"></param>
    /// <param name="b"></param>
    /// <param name="N"></param>
    void solveLinearEquation(double *A, double *b, size_t N);


    /// <summary>
    /// compute standard Euclidean norm - cublasDnrm2
    /// </summary>
    /// <param name="n"></param>
    /// <param name="x"></param>
    /// <param name="result"></param>
    void vectorNorm(int n, double *x, double *result);


    /// <summary>
    /// 
    /// </summary>
    /// <param name="n"></param>
    /// <param name="alpha"></param>
    /// <param name="x"></param>
    /// <param name="incx"></param>
    /// <param name="y"></param>
    /// <param name="incy"></param>
    void cublasAPIDaxpy(int n, const double *alpha, const double *x, int incx, double *y, int incy);

    /// <summary>
    /// Execution plan form computation of A *x = b equation . 
    /// A is a Sparse Tensor. Sparse Tensor.
    /// 
    /// QR solver with reorder "symrcm".
    /// </summary>
    /// <param name="csrRowInd"></param>
    /// <param name="csrColInd"></param>
    /// <param name="csrValInd"></param>
    /// <param name="b"></param>
    /// <param name="x"></param>
    void solverLinearEquationSP(int m, int n , int nnz, int *csrRowInd, int *csrColInd, double *csrValues, double *b, double *x, int* singularity);

    /// <summary>
    /// 
    /// </summary>
    /// <param name="m">IN</param>
    /// <param name="n">IN</param>
    /// <param name="nnz">IN</param>
    /// <param name="cooRowInd">IN/OUT sorted</param>
    /// <param name="cooColInd">IN/OUT sorted - csrColInd </param>
    /// <param name="cooValues">IN/OUT sorted (INPT) </param>
    /// <param name="csrRowInd">OUT vector - m + 1</param>
    /// <param name="INPT">IN/OUT vector[nnz], permutation from coo into csr</param>
    /// <param name="sort">if true INPT is not reused in computation</param>
    void transformToCsr(int m, int n, int nnz, int *cooRowInd, int *cooColInd, double *cooValues, int* csrRowInd, int *PT, bool sort);
        
    /// <summary>
    /// Invert addressing in PT permutation vector .
    /// </summary>
    /// <param name="n">vector length</param>
    /// <param name="PT">input PT permutation</param>
    /// <param name="INV">output inverse PT</param>
    void invertPermuts(int n, int *PT, int *INV);
    
    ///
    ~GPULinearSystem();

    private:

    /// <summary>
    /// Debug current stream execution state !
    /// </summary>
    void validateStreamState();


  private:
    cudaStream_t stream = nullptr;

    /// cuda variables
    cusolverDnHandle_t handle = nullptr;

    cusolverSpHandle_t cusolverSpHandle = nullptr;

    /// cusparse handle
    cusparseHandle_t cusparseHandle;

    cublasHandle_t cublasHandle = nullptr;

    cudaError_t lastError = cudaSuccess;

    /// cusolver context
    int Lwork = 0;

    /// additional workspace requirment imposed by LU solver
    double *Workspace = nullptr;

    /// lu factorization output vector pivot indices
    int *devIpiv = nullptr;

    /// data from Factorization or Solver
    int *devInfo = nullptr;

    /// pBuffer will store intermediate computation from Xcoosort functions
    void *pBuffer;

    /// actual allocation for Xcoosort
    size_t pBufferSizeInBytes;

    // first permutation vector, XcoosortByColumn
    int *PT1;

    // second permutation vector, XcoosortByRow
    int *PT2;

    size_t PT_nnz;

    /// cusolver tensor A default descriptor
    cusparseMatDescr_t descrA;
};

} // namespace solver

#endif //_LINEAR_SYSTEM_CUH_