#ifndef _TENSOR_OPERATION_H_
#define _TENSOR_OPERATION_H_

#include "cuda_runtime.h"

#include "cusparse.h"

class TensorOperation {
  public:
    /// Setup sparse handler
    TensorOperation(cudaStream_t stream);

    ~TensorOperation();

    /// <summary>
    /// 
    /// </summary>
    /// <param name="cooRowInd"></param>
    /// <param name="cooColInd"></param>
    /// <param name="cooVal"></param>
    /// <param name="nnz"></param>
    /// <param name="ld"></param>
    /// <param name="csrRowInd">OUTPUT - client must provide alloc size ( ld + 1 )</param>
    void convertToCSR(int *cooRowInd, int *cooColInd, double * cooVal, int nnz, int ld, int *csrRowInd);

  private:
         
    cudaStream_t stream;

    /// COO - cuSparse Handle - Coordinated Format conversion from COO into CSR direct
    cusparseHandle_t cuSparseHandle;

    cusparseStatus_t status;

    /// COO - cuSparse computation buffer size
    size_t pBufferSizeInBytes = 0;

    /// COO - cuSparse computation buffer
    void *pBuffer;

    /// permutation vector - zaporzyc z cudf <- Memory Manager spowalniac process realokacji !!!
    int *Prm;

    int Psize;
};

#endif _TENSOR_OPERATION_H_
