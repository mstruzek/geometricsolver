#ifndef _TENSOR_OPERATION_H_
#define _TENSOR_OPERATION_H_

#include "cuda_runtime_api.h"

#include "cublas_v2.h"              /// cuBlas
#include "cusparse.h"               /// cuSparse

#include "quda.h"

class TensorOperation {
  public:
    /// Setup sparse handler
    TensorOperation(cudaStream_t stream);

    ~TensorOperation();
        
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
    /// Set memory value of type CUDA_R32I ( by definition cuMemsetD2D32Async )
    /// </summary>
    /// <param name="devPtr"></param>
    /// <param name="value"></param>
    /// <param name="widht"></param>
    /// <param name="stream"></param>
    void memsetD32I(int *devPtr, int value, size_t size, cudaStream_t stream);

    /// <summary>
    /// Convert coo tensor format into csr format.
    /// </summary>
    /// <param name="m">IN</param>
    /// <param name="n">IN</param>
    /// <param name="nnz">IN</param>
    /// <param name="cooRowInd">IN/OUT sorted</param>
    /// <param name="cooColInd">IN/OUT sorted - csrColInd </param>
        /// <param name="csrRowInd">OUT vector - m + 1</param>
    /// <param name="INPT">IN/OUT vector[nnz], permutation from coo into csr</param>
    void convertToCsr(int m, int n, int nnz, int *cooRowInd, int *cooColInd, int *csrRowInd,
                        int *PT);

    /// <summary>
    /// Gather vector operation ; PT[.] = PT1[PT2[.]]
    /// </summary>
    /// <param name="nnz">non zero elements, length</param>
    /// <param name="PT1">input value vector</param>
    /// <param name="PT2">permutation vector</param>
    /// <param name="PT">output value vector</param>
    template <typename ValueType>
    void gatherVector(int nnz, cudaDataType valueType, ValueType *PT1, int *PT2, ValueType *PT);


    /// <summary>
    /// Invert addressing in PT permutation vector .
    /// </summary>
    /// <param name="n">vector length</param>
    /// <param name="PT">input PT permutation</param>
    /// <param name="INV">output inverse PT</param>
    void invertPermuts(int n, int *PT, int *INV);


    /// <summary>
    /// debug function. stdout device coo vectors.
    /// </summary>
    /// <param name="stream"></param>
    /// <param name="m"></param>
    /// <param name="n"></param>
    /// <param name="nnz"></param>
    /// <param name="d_cooRowInd"></param>
    /// <param name="d_cooColInd"></param>
    /// <param name="d_cooVal"></param>
    /// <param name="title"></param>
    void stdout_coo_vector(cudaStream_t stream, int m, int n, int nnz, int *d_cooRowInd, int *d_cooColInd, double *d_cooVal,
                                            const char *title);

    /// <summary>
    /// debug function. stdout coo vector in tensor layout.
    /// </summary>
    /// <param name="stream"></param>
    /// <param name="m"></param>
    /// <param name="n"></param>
    /// <param name="nnz"></param>
    /// <param name="d_cooRowInd"></param>
    /// <param name="d_cooColInd"></param>
    /// <param name="d_cooVal"></param>
    /// <param name="title"></param>
    void stdout_coo_tensor(cudaStream_t stream, int m, int n, int nnz, int *d_cooRowInd, int *d_cooColInd, double *d_cooVal,
                                            const char *title);

  private:
         
    cudaStream_t stream;

    /// COO - cuSparse Handle - Coordinated Format conversion from COO into CSR direct
    cusparseHandle_t sparseHandle;

    /// CUBLAS - sapxy operation , vector norm     
    cublasHandle_t cublasHandle;

    /// permutation vector - zaporzyc z cudf <- Memory Manager spowalniac process realokacji !!!
    utility::dev_vector<int> Prm;
    

    int Psize;


    /// pBuffer will store intermediate computation from Xcoosort functions
    utility::dev_vector<char> pBuffer;

    /// actual allocation for Xcoosort
    size_t pBufferSizeInBytes;

    // first permutation vector, XcoosortByColumn
    utility::dev_vector <int> PT1;

    // second permutation vector, XcoosortByRow
    utility::dev_vector<int>  PT2;

    size_t PT_nnz;

};




#endif _TENSOR_OPERATION_H_
