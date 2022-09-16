#include "format_encoder.h"

/// Thrust requirments

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>

#include "settings.h"


/// <summary>
///
/// </summary>
/// <param name="stream"></param>
FormatEncoder::FormatEncoder(cudaStream_t stream) : stream(stream) {

    /// set thrust::cuda stream of operations
    thrust::cuda::par.on(stream);

    ///  ------------------------------------------------------------------------------ ///
    cusparseStatus_t status;

    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        auto errorName = cusparseGetErrorName(status);
        auto errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[cusparse/error]  handler initialization failed ; ( %s ) %s \n", errorName, errorStr);
        exit(-1);
    }

    status = cusparseSetStream(handle, stream);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        auto errorName = cusparseGetErrorName(status);
        auto errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[gpu/cusparse] cusparse stream set failure ; ( %s ) %s \n", errorName, errorStr);
        exit(-1);
    }
}

/// --------------------------------------------------------------------------------------- ///

// StructeOfArray of array
typedef thrust::device_vector<int>::iterator cooRowIterator;
typedef thrust::device_vector<int>::iterator cooColIterator;
typedef thrust::device_vector<double>::iterator cooValIterator;

typedef thrust::tuple<int, int> cooCoordinate;
typedef thrust::tuple<int, int, double> objCoordinate;

typedef thrust::tuple<cooRowIterator, cooColIterator> cooIterator;
typedef thrust::tuple<cooRowIterator, cooColIterator, cooValIterator> objectIterator;

/// --------------------------------------------------------------------------------------- ///

/// --------------------------------------------------------------------------------------- ///

__device__ constexpr int ELEMENT_COO_ROW = 0;
__device__ constexpr int ELEMENT_COO_COLUMN = 1;

///
/// Predicate of Coordinate Equivalence
///
struct coordinate_equal {

    __device__ bool operator()(cooCoordinate left, cooCoordinate right) const {
        return left.get<ELEMENT_COO_ROW>() == right.get<ELEMENT_COO_ROW>() && left.get<ELEMENT_COO_COLUMN>() == right.get<ELEMENT_COO_COLUMN>();
    }
};

///
/// Coordinate Comparator ; sort by column and sort by column
///
struct coordinate_compare {
    __device__ bool operator()(cooCoordinate left, cooCoordinate right) {
        if (left.get<ELEMENT_COO_ROW>() < right.get<ELEMENT_COO_ROW>())
            return true;
        if (left.get<ELEMENT_COO_ROW>() == right.get<ELEMENT_COO_ROW>()) {
            if (left.get<ELEMENT_COO_COLUMN>() < right.get<ELEMENT_COO_COLUMN>())
                return true;       
        }
        return false;
    }
};

/// --------------------------------------------------------------------------------------- ///

/// <summary>
/// Compact COO journal format into CSR canonical form.
/// </summary>
/// <param name="nnz">[in] non-zero elements </param>
/// <param name="d_cooRowInd">[in]</param>
/// <param name="d_cooColIndA">[in]</param>
/// <param name="d_cooVal">[in]</param>
/// <param name="d_cooRowInd_order">[out] ordered indcies coo</param>
/// <param name="d_csrRowInd">[out]</param>
/// <param name="d_csrColIndA">[out]</param>
/// <param name="d_csrVal">[out]</param>
/// <param name="onnz">[out] non-zero elements after reduction</param>
void FormatEncoder::compactToCsr(int nnz, int m, int *d_cooRowIndA, int *d_cooColIndA, double *d_cooValA, int* d_cooRowInd_order, int *d_csrRowPtr, int *d_csrColInd,
                                 double *d_csrVal,
                                 int &onnz) {

    /// intput iteratory - no copy COO journal format

    thrust::device_ptr<int> d_cooRowPtrA_begin = thrust::device_ptr<int>(d_cooRowIndA);
    thrust::device_ptr<int> d_cooRowPtrA_end = thrust::device_ptr<int>(&d_cooRowIndA[nnz]);

    thrust::device_ptr<int> d_cooColIndA_begin = thrust::device_ptr<int>(d_cooColIndA);
    thrust::device_ptr<int> d_cooColIndA_end = thrust::device_ptr<int>(&d_cooColIndA[nnz]);

    thrust::device_ptr<double> d_cooVal_begin = thrust::device_ptr<double>(d_cooValA);
    thrust::device_ptr<double> d_cooVal_end = thrust::device_ptr<double>(&d_cooValA[nnz]);

    /// output iterators - no copy COO compact form
    thrust::device_ptr<int> o_cooRowPtrA_begin = thrust::device_ptr<int>(d_cooRowInd_order);
    thrust::device_ptr<int> o_cooRowPtrA_end = thrust::device_ptr<int>(&d_cooRowInd_order[nnz]);

    thrust::device_ptr<int> o_cooColIndA_begin = thrust::device_ptr<int>(d_csrColInd);
    thrust::device_ptr<int> o_cooColIndA_end = thrust::device_ptr<int>(&d_csrColInd[nnz]);

    thrust::device_ptr<double> o_cooVal_begin = thrust::device_ptr<double>(d_csrVal);
    thrust::device_ptr<double> o_cooVal_end = thrust::device_ptr<double>(&d_csrVal[nnz]);

    /// first sort ( ROW, COL ) pairs

    thrust::sort_by_key(thrust::device, 
        thrust::zip_iterator<cooIterator>(thrust::make_tuple(d_cooRowPtrA_begin, d_cooColIndA_begin)),
        thrust::zip_iterator<cooIterator>(thrust::make_tuple(d_cooRowPtrA_end, d_cooColIndA_end)), d_cooVal_begin, 
        coordinate_compare());

    /// reduce adjecent structures 
    /// 
    /// ADD adjecent (ROW,COL) pairs  - commands "add" , "set"
    /// 

    auto last = thrust::reduce_by_key(thrust::device, 
        thrust::zip_iterator<cooIterator>(thrust::make_tuple(d_cooRowPtrA_begin, d_cooColIndA_begin)),
        thrust::zip_iterator<cooIterator>(thrust::make_tuple(d_cooRowPtrA_end, d_cooColIndA_end)), d_cooVal_begin,
        thrust::zip_iterator<cooIterator>(thrust::make_tuple(o_cooRowPtrA_begin, o_cooColIndA_begin)), o_cooVal_begin,
        coordinate_equal(), thrust::plus<double>());

    /// Await

    /// compressed COO format
    size_t nnz_c = std::distance(o_cooVal_begin, last.second);
 
    onnz = nnz_c;
    
    /// ------------------------------------------------------------------------------ ///
        
    cusparseStatus_t status;
    /// create csrRowPtrA    -  async execution
    status = cusparseXcoo2csr(handle, d_cooRowInd_order, nnz_c, m, d_csrRowPtr, CUSPARSE_INDEX_BASE_ZERO);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        auto errorName = cusparseGetErrorName(status);
        auto errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[cusparse/error]  conversion to csr storage format ; ( %s ) %s \n", errorName, errorStr);
        exit(1);
    }

    /// ------------------------------------------------------------------------------ ///

    if (settings::DEBUG) {
        fprintf(stderr, "[encoder.csr] tensor representation in CSR done !\n");
    }
}

/// --------------------------------------------------------------------------------------- ///

FormatEncoder::~FormatEncoder() {
    cusparseStatus_t status;
    /// Thread Safe ?????
    status = cusparseDestroy(handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        auto errorName = cusparseGetErrorName(status);
        auto errorStr = cusparseGetErrorString(status);
        fprintf(stderr, "[cusparse/error]  uninitialization failure ; ( %s ) %s \n", errorName, errorStr);
        exit(-1);
    }
}