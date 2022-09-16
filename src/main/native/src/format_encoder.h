#pragma once

#include "cuda_runtime_api.h"

#include "cusparse.h"

class FormatEncoder {

  public:
    /// <summary>
    ///
    /// </summary>
    /// <param name="stream"></param>
    FormatEncoder(cudaStream_t stream);

    /// <summary>
    /// Compact COO journal format into CSR canonical form.
    /// </summary>
    /// <param name="nnz">[in] non-zero elements </param>
    /// <param name="m">[in] number of rows </param>
    /// <param name="d_cooRowInd">[in]</param>
    /// <param name="d_cooColIndA">[in]</param>
    /// <param name="d_cooVal">[in]</param>
    /// <param name="d_csrRowInd">[out]</param>
    /// <param name="d_csrColIndA">[out]</param>
    /// <param name="d_csrVal">[out]</param>
    /// <param name="onnz">[out] non-zero elements after reduction</param>
    void compactToCsr(int nnz, int m, int *d_cooRowIndA, int *d_cooColIndA, double *d_cooValA, int *d_csrRowPtr, int *d_csrColInd, double *d_csrVal, int &onnz);

    ~FormatEncoder();    

  private:
    cudaStream_t stream;

    cusparseHandle_t handle;
};
