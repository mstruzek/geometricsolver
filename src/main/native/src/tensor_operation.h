#ifndef _TENSOR_OPERATION_H_
#define _TENSOR_OPERATION_H_

#include "cuda_runtime.h"

#include "cublas_v2.h"              /// cuBlas
#include "cusparse.h"               /// cuSparse

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
    /// Convert coo tensor format into csr format.
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
    void convertToCsr(int m, int n, int nnz, int *cooRowInd, int *cooColInd, double *cooValues, int *csrRowInd,
                        int *PT, bool sort);

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


  private:
    /// <summary>
    /// Debug current stream execution state !
    /// </summary>
    void validateStreamState();

  private:
         
    cudaStream_t stream;

    /// COO - cuSparse Handle - Coordinated Format conversion from COO into CSR direct
    cusparseHandle_t sparseHandle;

    /// CUBLAS - sapxy operation , vector norm     
    cublasHandle_t cublasHandle;

    /// permutation vector - zaporzyc z cudf <- Memory Manager spowalniac process realokacji !!!
    int *Prm;

    int Psize;


    /// pBuffer will store intermediate computation from Xcoosort functions
    void *pBuffer;

    /// actual allocation for Xcoosort
    size_t pBufferSizeInBytes;

    // first permutation vector, XcoosortByColumn
    int *PT1;

    // second permutation vector, XcoosortByRow
    int *PT2;

    size_t PT_nnz;

};

#endif _TENSOR_OPERATION_H_
