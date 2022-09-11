#ifndef _TENSOR_LAYOUT_CUH_
#define _TENSOR_LAYOUT_CUH_

#include "device_launch_parameters.h"

#include "math.h"

#include "model.cuh"

#ifndef __GPU_COMM_INL__
#define __GPU_COMM_INL__ __host__ __device__
#endif

#ifndef __GPU_DEV_INL__
#define __GPU_DEV_INL__ __device__
#endif

#ifndef __GPU_DEV_INLF__
#define __GPU_DEV_INLF__ __forceinline__ __device__
#endif



#define DEGREES_TO_RADIANS 0.017453292519943295;

__GPU_DEV_INLF__ double toRadians(double angdeg) { return angdeg * DEGREES_TO_RADIANS; }


namespace graph {

class Vector;

//=================================================================================

// 2x2 small grid
class BlockLayout {
  public:
    __GPU_DEV_INLF__ BlockLayout() {}

    __GPU_DEV_INLF__ void set(int row, int col, double value) {
        if (row < ld && col < cols) {
            tensor[ld * row + col] = value;
        }
    }

    __GPU_DEV_INLF__ double get(int row, int col) const {
        if (row < ld && col < cols) {
            return tensor[ld * row + col];
        }
        return NAN;
    }

  private:
    double tensor[4] = {0.0};
    const int ld = 2;
    const int cols = 2;
};
//=================================================================================

//=================================================================================

class DenseLayout {

  public:
    __GPU_DEV_INLF__ DenseLayout() : ld(0), rowOffset(0), colOffset(0), m_A(NULL) {}

    __GPU_DEV_INLF__ DenseLayout(size_t _ld, size_t _rowOffset, size_t _colOffset, double *A)
        : ld(_ld), rowOffset(_rowOffset), colOffset(_colOffset), m_A(A) {}

    __GPU_DEV_INLF__ void set(int row, int col, double value) {
        ///
        m_A[ld * (colOffset + col) + rowOffset + row] = value;
    }

    __GPU_DEV_INLF__ void add(int row, int col, double value) {
        ///
        m_A[ld * (colOffset + col) + rowOffset + row] += value;
    }

    __GPU_DEV_INLF__ double get(int row, int col) const {
        ///
        return m_A[ld * (colOffset + col) + rowOffset + row];
    }

  public:
    // leading dimension
    const size_t ld;

    // row offset
    const size_t rowOffset;

    // column offset
    const size_t colOffset;

    // device storage
    double *const m_A;
};

//=================================================================================

class SparseLayout {
  public:
    __GPU_DEV_INLF__ SparseLayout(int *_accWriteOffset, int *_cooRowInd, int *_cooColInd, double *_cooVal)
        : accWriteOffset(_accWriteOffset), cooRowInd(_cooRowInd), cooColInd(_cooColInd), cooVal(_cooVal) {}

    __GPU_DEV_INLF__ void set(int row, int col, double value) {
        /// standard 128 byte cache line - no additional contention in a warp
        int offset = accWriteOffset[blockIdx.x * blockDim.x + threadIdx.x];
        int nextOffset = accWriteOffset[blockIdx.x * blockDim.x + threadIdx.x + 1]; /// uzupelnic skladowa
        int blockSize = nextOffset - offset;

        for (int t = 0; t < blockSize; ++t) {
            const int row_at = cooRowInd[offset + t];
            const int col_at = cooColInd[offset + t];
            if (row_at == -1 && col_at == -1) {
                /// SET VALUE
                cooRowInd[offset + t] = row;
                cooColInd[offset + t] = col;
                cooVal[offset + t] = value;
                return;
            }
        }
    }

    __GPU_DEV_INLF__ void add(int row, int col, double value) {
        /// 128 byte cache line - contention in a warp
        int offset = accWriteOffset[blockIdx.x * blockDim.x + threadIdx.x];
        int nextOffset = accWriteOffset[blockIdx.x * blockDim.x + threadIdx.x + 1]; /// uzupelnic skladowa
        int blockSize = nextOffset - offset;

        for (int t = 0; t < blockSize; ++t) {
            const int row_at = cooRowInd[offset + t];
            const int col_at = cooColInd[offset + t];
            if (row_at == row && col_at == col) {
                /// override value
                cooVal[offset + t] += value;
                return;
            } else if (row_at == -1 && col_at == -1) {
                /// SET VALUE
                cooRowInd[offset + t] = row;
                cooColInd[offset + t] = col;
                cooVal[offset + t] = value;
                return;
            }
        }
    }

    __GPU_DEV_INLF__ double get(int row, int col) const {
        int offset = accWriteOffset[blockIdx.x * blockDim.x + threadIdx.x];
        int nextOffset = accWriteOffset[blockIdx.x * blockDim.x + threadIdx.x + 1];
        int blockSize = nextOffset - offset;

        for (int t = 0; t < blockSize; ++t) {
            if (cooRowInd[offset + t] == row && cooColInd[offset + t] == col) {
                return cooVal[offset + t];
            }
        }
        return 0.0; /// invalidate computation state
    }

  private:
    int *const accWriteOffset; /// cub::exclusive_scan , offset for this Constraint or Geometric Block

    int *const cooRowInd; /// COO row indicies
    int *const cooColInd; /// COO column indicies
    double *const cooVal; /// COO values
};

//=================================================================================

/// Local Thread Index Storage
template <typename DataType> class BlockIterator {
  public:
    __GPU_DEV_INLF__ BlockIterator() {

        /// 128 byte cache line
        extern __shared__ DataType kernel_shared[];

        /// kernel absolute owning thread access id
        threadId = blockIdx.x * blockDim.x + threadIdx.x;

        /// shared buffer of persmissions/slots
        slot = kernel_shared;
    }

    __GPU_DEV_INLF__ void reset(DataType value = DataType()) { slot[threadIdx.x] = value; }

    __GPU_DEV_INLF__ DataType next() {
        DataType value = slot[threadIdx.x];
        ++slot[threadIdx.x];
        return value;
    }

    __GPU_DEV_INLF__ DataType value() {
        DataType value = slot[threadIdx.x];
        return value;
    }

    /// kernel absolute threadId
    __GPU_DEV_INLF__ int thread_id() { return threadId; }

  private:
    /// kernel shared reference
    DataType *slot;

    /// one owning slot for each kernel thread
    int threadId;
};

//=================================================================================

/// Direct Access - no permission checks !
///
class DirectSparseLayout {
  public:
    __GPU_DEV_INLF__ DirectSparseLayout() : accOffset(NULL), P(NULL), cooRowInd(NULL), cooColInd(NULL), cooVal(NULL) {}

    __GPU_DEV_INLF__ DirectSparseLayout(int *const _accOffset, int *const _P, int *_cooRowInd, int *_cooColInd,
                                        double *_cooVal)
        : accOffset(_accOffset), P(_P), cooRowInd(_cooRowInd), cooColInd(_cooColInd), cooVal(_cooVal) {
        iterator.reset();
    }

    __GPU_DEV_INLF__ void set(int row, int col, double value) {
        unsigned threadId = iterator.thread_id();
        unsigned offset = iterator.next();
        unsigned threadOffset = accOffset[threadId];

        /// override value - inversed indicies
        cooVal[P[threadOffset + offset]] = value;
    }

    __GPU_DEV_INLF__ void add(int row, int col, double value) {
        unsigned threadId = iterator.thread_id();
        unsigned threadOffset = accOffset[threadId];
        unsigned itr_offset = iterator.value();

        // ADD
        for (unsigned itr = 0; itr < itr_offset; itr++) {
            int at_row = cooRowInd[threadOffset + itr];
            int at_col = cooColInd[threadOffset + itr];
            if (at_row == row && at_col == col) {

                cooVal[P[threadOffset + itr]] += value;
                return;
            } else if (at_row == -1 && at_col == -1) {

                break;
            }
        }

        // SET
        unsigned offset = iterator.next();
        /// override value - inversed indicies
        cooVal[P[threadOffset + offset]] = value;
    }

    __GPU_DEV_INLF__ double get(int row, int col) const {
        /// invalidate computation state
        /// !!!!!!!!! ERROR ---
        return NAN;
    }

  private:
    BlockIterator<int> iterator{};

  private:
    int *const accOffset; /// thread acc offset
    int *const P;         /// direct indicies dense vector

    double *const cooVal; /// COO values

#define __BEFORE_TRANSFORMATION__
    int *cooRowInd; /// COO row not-transformed
    int *cooColInd; /// COO col not-transformed
};

//=================================================================================

template <typename LLayout = BlockLayout> class Tensor {
  public:
    __GPU_DEV_INLF__ Tensor(LLayout const &_u = LLayout(), bool _intention = false) : u(_u), intention(_intention) {}

    __GPU_DEV_INLF__ Tensor(double a00, double a01, double a10, double a11) : Tensor(LLayout()) {
        u.set(0, 0, a00);
        u.set(0, 1, a01);
        u.set(1, 0, a10);
        u.set(1, 1, a11);
    }

    __GPU_DEV_INLF__ void setVector(int offsetRow, int offsetCol, graph::Vector const &value) {
        /// vertical
        if (intention) { // tensor "B"
            u.set(offsetRow + 0, offsetCol, getVectorX(value));
            u.set(offsetRow + 1, offsetCol, getVectorY(value));
        } else { // tensor "Jacobian"
            u.set(offsetRow, offsetCol + 0, getVectorX(value));
            u.set(offsetRow, offsetCol + 1, getVectorY(value));
        }
    };

    __GPU_DEV_INLF__ void setValue(int offsetRow, int offsetCol, double const &value) {
        u.set(offsetRow, offsetCol, value);
    }

    __GPU_DEV_INL__ double getValue(int offsetRow, int offsetCol) const { return u.get(offsetRow, offsetCol); }

    __GPU_DEV_INLF__ void plusSubTensor(int offsetRow, int offsetCol, Tensor<graph::BlockLayout> const &mt) {
        /// small tensor
        double a00 = mt.u.get(0, 0);
        double a01 = mt.u.get(0, 1);
        double a10 = mt.u.get(1, 0);
        double a11 = mt.u.get(1, 1);

        u.add(offsetRow + 0, offsetCol, a00);
        u.add(offsetRow + 1, offsetCol, a10);
        u.add(offsetRow + 0, offsetCol + 1, a01);
        u.add(offsetRow + 1, offsetCol + 1, a11);
    }

    __GPU_DEV_INLF__ void setSubTensor(int offsetRow, int offsetCol, Tensor<graph::BlockLayout> const &mt) {
        /// small tensor
        double a00 = mt.u.get(0, 0);
        double a01 = mt.u.get(0, 1);
        double a10 = mt.u.get(1, 0);
        double a11 = mt.u.get(1, 1);

        u.set(offsetRow + 0, offsetCol, a00);
        u.set(offsetRow + 1, offsetCol, a10);
        u.set(offsetRow + 0, offsetCol + 1, a01);
        u.set(offsetRow + 1, offsetCol + 1, a11);
    }

    __GPU_DEV_INLF__ Tensor<graph::BlockLayout> multiplyC(double scalar) {
        if constexpr (std::is_same_v<LLayout, graph::BlockLayout>) {
            /// SmallTensor
            double a00 = u.get(0, 0) * scalar;
            double a01 = u.get(0, 1) * scalar;
            double a10 = u.get(1, 0) * scalar;
            double a11 = u.get(1, 1) * scalar;
            return Tensor<graph::BlockLayout>(a00, a01, a10, a11);
        }
        return Tensor();
    }

    __GPU_DEV_INLF__ Tensor<graph::BlockLayout> plus(Tensor<graph::BlockLayout> const &other) {
        if constexpr (std::is_same_v<LLayout, graph::BlockLayout>) {
            /// SmallTensor
            double a00 = u.get(0, 0) + other.u.get(0, 0);
            double a01 = u.get(0, 1) + other.u.get(0, 1);
            double a10 = u.get(1, 0) + other.u.get(1, 0);
            double a11 = u.get(1, 1) + other.u.get(1, 1);
            return Tensor<graph::BlockLayout>(a00, a01, a10, a11);
        }
        return Tensor();
    }

    __GPU_DEV_INLF__ Tensor<graph::BlockLayout> transpose() const {
        if constexpr (std::is_same_v<LLayout, graph::BlockLayout>) {

            double a00 = u.get(0, 0);
            double a01 = u.get(0, 1);
            double a10 = u.get(1, 0);
            double a11 = u.get(1, 1);
            return Tensor<graph::BlockLayout>(a00, a10, a01, a11);
        }
        return Tensor();
    }

    friend class Tensor<graph::BlockLayout>;
    friend class Tensor<graph::SparseLayout>;
    friend class Tensor<graph::DenseLayout>;
    friend class Tensor<graph::DirectSparseLayout>;

  private:
    bool intention; // vector put operation vertical if true / horizontal otherwise
    LLayout u;
};

//=================================================================================

__GPU_DEV_INLF__ Tensor<DenseLayout> tensorDevMem(DenseLayout parent, int rowOffset, int colOffset,
                                                  bool intention = true) {
    DenseLayout layout(parent.ld, parent.rowOffset + rowOffset, parent.colOffset + colOffset, parent.m_A);
    Tensor<DenseLayout> tensor(layout, intention);
    return tensor;
}

__GPU_DEV_INLF__ Tensor<SparseLayout> tensorDevMem(SparseLayout layout, bool intention = true) {
    Tensor<SparseLayout> tensor(layout, intention);
    return tensor;
}

__GPU_DEV_INLF__ Tensor<DirectSparseLayout> tensorDevMem(DirectSparseLayout layout, bool intention = true) {
    Tensor<DirectSparseLayout> tensor(layout, intention);
    return tensor;
}

__GPU_DEV_INLF__ Tensor<BlockLayout> make_block_tensor(double a00, double a01, double a10, double a11) {
    return Tensor<graph::BlockLayout>(a00, a10, a01, a11);
}

//=================================================================================

/// 2x2
class SmallTensor {
  public:
    __GPU_DEV_INLF__ static Tensor<BlockLayout> tensorR() {
        double a00 = 0.0;
        double a01 = -1.0;
        double a10 = 1.0;
        double a11 = 0.0;
        return Tensor<BlockLayout>(a00, a01, a10, a11);
    }

    __GPU_DEV_INLF__ static Tensor<BlockLayout> rotation(double alfa) {
        double radians = toRadians(alfa);
        double a00 = cos(radians);
        double a01 = -1.0 * sin(radians);
        double a10 = sin(radians);
        double a11 = cos(radians);
        return Tensor<BlockLayout>(a00, a01, a10, a11);
    }

    __GPU_DEV_INLF__ static Tensor<BlockLayout> identity(double value) {
        return Tensor<BlockLayout>(value, 0.0, 0.0, value);
    }

    __GPU_DEV_INLF__ static Tensor<BlockLayout> diagonal(double diagonal) {
        return Tensor<BlockLayout>(diagonal, 0.0, 0.0, diagonal);
    }
};

//=================================================================================

typedef Tensor<BlockLayout> TensorBlock;

///
/// Transpose Tensor Adapter - utility class for transposed storage operations into output tensor
///  -- transpose/inversed offsetRow/offsetCol order
///
template <typename LLayout> class AdapterTensor : public Tensor<LLayout> {

  public:
    __GPU_DEV_INLF__ AdapterTensor(LLayout layout, bool intention) : Tensor<LLayout>(layout, intention) {}

    __GPU_DEV_INLF__ void setVector(int offsetRow, int offsetCol, Vector const &value) {
        Tensor<LLayout>::setValue(offsetCol + 0, offsetRow, getVectorX(value));
        Tensor<LLayout>::setValue(offsetCol + 1, offsetRow, getVectorY(value));
    }

    __GPU_DEV_INLF__ void setValue(int offsetRow, int offsetCol, double const &value) {
        Tensor<LLayout>::setValue(offsetCol, offsetRow, value);
    }

    __GPU_DEV_INLF__ void setSubTensor(int offsetRow, int offsetCol, Tensor<graph::BlockLayout> const &mt) {
        Tensor<LLayout>::plusSubTensor(offsetCol, offsetRow, mt.transpose());
    }
};

// transponsed operations
template <typename LLayout>
__GPU_DEV_INLF__ static AdapterTensor<LLayout> transposeTensorDevMem(LLayout parent, bool intention) {
    AdapterTensor<LLayout> t{parent, intention};
    return t;
}

//=================================================================================



} // namespace graph

#endif // _TENSOR_LAYOUT_CUH_