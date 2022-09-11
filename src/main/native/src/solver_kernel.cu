#include "solver_kernel.cuh"

#include "kernel_traits.h"

#ifndef DEFAULT_BLOCK_DIM
#define DEFAULT_BLOCK_DIM 512
#endif

#ifndef OBJECTS_PER_THREAD
#define OBJECTS_PER_THREAD 1
#endif

#define TENSOR_SPARSE_LAYOUT

/// =======================================================================================
/// <summary>
/// Kernel Permuatation compaction routine  PT[i] = P1[P2[i]] .
/// </summary>
/// </summary>
/// <param name="nnz"></param>
/// <param name="PT1"></param>
/// <param name="PT2"></param>
/// <param name="PT"></param>
/// <returns></returns>
__global__ void __compactPermutationVector__(int nnz, int *PT1, int *PT2, int *PT) {

    int tId = blockDim.x * blockIdx.x + threadIdx.x;

#define ROUTINE_ELEMENT_RANGE 4

    if (threadIdx.x < nnz) {
        int indicie = PT2[tId];
        PT[tId] = PT1[indicie];
    }
}

/// =======================================================================================
/// <summary>
///  Kernel Permutation compaction routine. Kernel Executor.  PT[i] = PT1[PT2[i]] .
/// </summary>
/// <param name="K_gridDim"></param>
/// <param name="K_blockDim"></param>
/// <param name="K_stream"></param>
/// <param name="nnz"></param>
/// <param name="PT1"></param>
/// <param name="PT2"></param>
/// <param name="PT"></param>
void compactPermutationVector(unsigned K_gridDim, unsigned K_blockDim, cudaStream_t K_stream, int nnz, int *PT1,
                              int *PT2, int *PT) {

    __compactPermutationVector__<<<K_gridDim, K_blockDim, 0, K_stream>>>(nnz, PT1, PT2, PT);
}

/// =======================================================================================
/// <summary>
/// Inverse COO INP map - This is Direct Form !
/// </summary>
/// <param name="INP">dense INP vector - result of cusparseXcoosortByRow</param>
/// <param name="OUTP">dense output vector - direct form</param>
/// <param name="N">size of intput/output vector</param>
/// <returns></returns>
__global__ void __inversePermuationVector__(int *INP, int *OUTP, size_t N) {
    const unsigned threadId = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned offset = ELEMENTS_PER_THREAD * threadId;
    const unsigned upperLimit = offset + ELEMENTS_PER_THREAD;

    if (upperLimit < N) {

///  ?????   warning C4068: nieznana pragma �unroll� nvcc
///
#pragma unroll
        for (int T = 0; T < ELEMENTS_PER_THREAD; ++T) {
            OUTP[INP[offset + T]] = offset + T;
        }
    } else {
        const unsigned remainder = N - offset;
        for (int T = 0; T < remainder; ++T) {
            if (offset + T < N) {
                OUTP[INP[offset + T]] = offset + T;
            }
        }
    }
}

/// =======================================================================================
/// <summary>
/// Inverse COO indicies map ;  this is Direct Form !
/// </summary>
/// <param name="K_gridDim"></param>
/// <param name="K_blockDim"></param>
/// <param name="K_stream"></param>
/// <param name="INP">indicies desne in vector</param>
/// <param name="OUTP">inverse dense out vector - direct form</param>
/// <param name="N">size of intput/output vector</param>
/// <returns></returns>
void inversePermutationVector(unsigned K_gridDim, unsigned K_blockDim, cudaStream_t K_stream, int *INP, int *OUTP,
                              size_t N) {
    ///
    __inversePermuationVector__<<<K_gridDim, K_blockDim, 0, K_stream>>>(INP, OUTP, N);
}

/// =======================================================================================
///                             debug utility
/// =======================================================================================

__device__ constexpr const char *WIDEN_DOUBLE_STR_FORMAT = "%26d";
__device__ constexpr const char *FORMAT_STR_DOUBLE = " %11.2e";
__device__ constexpr const char *FORMAT_STR_IDX_DOUBLE = "%% %2d  %11.2e \n";
__device__ constexpr const char *FORMAT_STR_IDX_DOUBLE_E = "%%     %11.2e \n";
__device__ constexpr const char *FORMAT_STR_DOUBLE_CM = ", %11.2e";

template <typename... Args> __device__ void log(const char *formatStr, Args... args) {
    ///
    ::printf(formatStr, args...);
}

template <typename... Args> __device__ void log_error(const char *formatStr, Args... args) {
    ///
    ::printf(formatStr, args...);
}

///  ======================================================================================
/// <summary>
/// [debug] stdout tensor A - dense form.
/// </summary>
/// <param name="ecdata"></param>
/// <param name="rows"></param>
/// <param name="cols"></param>
/// <returns></returns>
__global__ void __stdoutTensorData__(ComputationState *ecdata, size_t rows, size_t cols) {

    ComputationState *ev = static_cast<ComputationState *>(ecdata);

    const graph::DenseLayout layout = graph::DenseLayout(ev->dimension, 0, 0, ev->A);
    const graph::Tensor<graph::DenseLayout> A = graph::tensorDevMem(layout, 0, 0);

    log("A \n");
    log("\n MatrixDouble2 - %lu x %lu **************************************** \n", rows, cols);

    /// table header
    for (int i = 0; i < cols / 2; i++) {
        log(WIDEN_DOUBLE_STR_FORMAT, i);
    }
    log("\n");

    /// table ecdata

    for (int i = 0; i < rows; i++) {
        log(FORMAT_STR_DOUBLE, A.getValue(i, 0));

        for (int j = 1; j < cols; j++) {
            log(FORMAT_STR_DOUBLE_CM, A.getValue(i, j));
        }
        if (i < cols - 1)
            log("\n");
    }
}

/// =======================================================================================
/// <summary>
/// [debug] stdout tensor A - dense form.
/// </summary>
/// <param name="stream"></param>
/// <param name="ecdata"></param>
/// <param name="rows"></param>
/// <param name="cols"></param>
/// <returns></returns>
KERNEL_EXECUTOR void stdoutTensorData(cudaStream_t stream, ComputationState *ecdata, size_t rows, size_t cols) {

    const unsigned GRID_DIM = 1;
    const unsigned BLOCK_DIM = 1;
    const unsigned NS = 0;

    __stdoutTensorData__<<<GRID_DIM, BLOCK_DIM, NS, stream>>>(ecdata, rows, cols);
}

/// =======================================================================================

__global__ void __stdoutRightHandSide__(ComputationState *ecdata, size_t rows) {

    ComputationState *ev = static_cast<ComputationState *>(ecdata);

    const graph::DenseLayout layout = graph::DenseLayout(rows, 0, 0, ev->b);
    const graph::Tensor<graph::DenseLayout> B = graph::tensorDevMem(layout, 0, 0);

    log("\n B \n");
    log("\n MatrixDouble1 - %lu x 1 ****************************************\n", rows);
    log("\n");
    /// table ecdata

    for (int i = 0; i < rows; i++) {
        log(FORMAT_STR_DOUBLE, B.getValue(i, 0));
        log("\n");
    }
}

/// =======================================================================================
/// <summary>
/// [debug] right hand side tensor B
/// </summary>
/// <param name="stream"></param>
/// <param name="ecdata"></param>
/// <param name="rows"></param>
/// <returns></returns>
KERNEL_EXECUTOR void stdoutRightHandSide(cudaStream_t stream, ComputationState *ecdata, size_t rows) {

    const unsigned GRID_DIM = 1;
    const unsigned BLOCK_DIM = 1;
    const unsigned NS = 0;

    __stdoutRightHandSide__<<<GRID_DIM, BLOCK_DIM, NS, stream>>>(ecdata, rows);
}

/// =======================================================================================
/// <summary>
/// [debug] State Vector print from device
/// </summary>
/// <param name="ecdata"></param>
/// <param name="rows"></param>
/// <returns></returns>
__global__ void __stdoutStateVector__(ComputationState *ecdata, size_t rows) {

    ComputationState *ev = static_cast<ComputationState *>(ecdata);

    const graph::DenseLayout layout = graph::DenseLayout(rows, 0, 0, ev->SV);
    const graph::Tensor<graph::DenseLayout> SV = graph::tensorDevMem(layout, 0, 0);

    log("\n SV - computation ( %d ) \n", ev->cID);
    log("\n MatrixDouble1 - %lu x 1 ****************************************\n", rows);
    log("\n");
    /// table ecdata

    for (int i = 0; i < rows; i++) {
        int pointId = ev->points[i / 2].id;
        double value = SV.getValue(i, 0);
        switch (i % 2 == 0) {
        case 0:
            log(FORMAT_STR_IDX_DOUBLE, pointId, value);
            break;
        case 1:
            log(FORMAT_STR_IDX_DOUBLE_E, value);
            break;
        }
        log("\n");
    }
}

/// =======================================================================================
/// <summary>
/// [debug] State Vector print from device
/// </summary>
/// <param name="stream"></param>
/// <param name="ecdata"></param>
/// <param name="rows"></param>
/// <returns></returns>
KERNEL_EXECUTOR void stdoutStateVector(cudaStream_t stream, ComputationState *ecdata, size_t rows) {

    const unsigned GRID_DIM = 1;
    const unsigned BLOCK_DIM = 1;
    const unsigned NS = 0;

    __stdoutStateVector__<<<GRID_DIM, BLOCK_DIM, NS, stream>>>(ecdata, rows);
}

///  ======================================================================================

/// --------------- [ KERNEL# GPU ]

__global__ void __CopyIntoStateVector__(double *SV, graph::Point *points, size_t size) {
    const unsigned threadId = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned offset = threadId * ELEMENTS_PER_THREAD;
    const unsigned upper_limit = offset + ELEMENTS_PER_THREAD;

    if (upper_limit < size) {

#pragma unroll
        for (int T = 0; T < ELEMENTS_PER_THREAD; ++T) {
            const unsigned IDX = offset + T;
            SV[2 * IDX + 0] = points[IDX].x;
            SV[2 * IDX + 1] = points[IDX].y;
        }

    } else {
        const unsigned REMAINDER = size - offset;

        for (int T = 0; T < REMAINDER; ++T) {
            const unsigned IDX = offset + T;
            if (IDX < size) {
                SV[2 * IDX + 0] = points[IDX].x;
                SV[2 * IDX + 1] = points[IDX].y;
            }
        }
    }
}

/// =======================================================================================
/// <summary>
/// Initialize State vector from actual points (without point id).
/// Reference mapping exists in Evaluation.pointOffset
/// </summary>
/// <param name="ec"></param>
/// <returns></returns>
KERNEL_EXECUTOR void CopyIntoStateVector(cudaStream_t stream, double *SV, graph::Point *points, size_t size) {

    typedef KernelTraits<ELEMENTS_PER_THREAD, DEFAULT_BLOCK_DIM> PointKernelTraits_t;

    const PointKernelTraits_t PointKernelTraits(size);
    const unsigned GRID_DIM = PointKernelTraits.GRID_DIM;
    const unsigned BLOCK_DIM = PointKernelTraits.BLOCK_DIM;

    __CopyIntoStateVector__<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(SV, points, size);
}

/// =======================================================================================
/// <summary>
/// Move computed position from State Vector into corresponding point object.
/// </summary>
/// <param name="ec"></param>
/// <returns></returns>
///
/// amortyzacja wzgledem inicjalizacji kernel a rejestrem watku
__global__ void __CopyFromStateVector__(graph::Point *points, double *SV, size_t size) {

    const unsigned threadId = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned offset = threadId * ELEMENTS_PER_THREAD;
    const unsigned upper_limit = offset + ELEMENTS_PER_THREAD;

    if (upper_limit < size) { ///  standard case

#pragma unroll
        for (int T = 0; T < ELEMENTS_PER_THREAD; ++T) {
            const unsigned IDX = offset + T;
            graph::Point *point = &points[IDX];
            point->x = SV[2 * IDX + 0];
            point->y = SV[2 * IDX + 1];
        }
    } else {
        const unsigned REMAINDER = size - offset;

        for (int T = 0; T < REMAINDER; ++T) {
            const unsigned IDX = offset + T;
            if (IDX < size) {
                graph::Point *point = &points[IDX];
                point->x = SV[2 * IDX + 0];
                point->y = SV[2 * IDX + 1];
            }
        }
    }
}

/// =======================================================================================
/// <summary>
/// Move computed position from State Vector into corresponding point object.
/// </summary>
/// <param name="stream"></param>
/// <param name="points"></param>
/// <param name="SV"></param>
/// <param name="size"></param>
/// <returns></returns>
KERNEL_EXECUTOR void CopyFromStateVector(cudaStream_t stream, graph::Point *points, double *SV, size_t size) {

    typedef KernelTraits<ELEMENTS_PER_THREAD, DEFAULT_BLOCK_DIM> PointKernelTraits_t;

    const PointKernelTraits_t PointKernelTraits(size);
    const unsigned GRID_DIM = PointKernelTraits.GRID_DIM;
    const unsigned BLOCK_DIM = PointKernelTraits.BLOCK_DIM;

    __CopyFromStateVector__<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(points, SV, size);
}

/// =======================================================================================
/// <summary>
/// Accumulate difference from newton-raphson method;  SV[] = SV[] + dx;
/// </summary>
/// <param name="SV"></param>
/// <param name="dx"></param>
/// <param name="N"></param>
/// <returns></returns>
__global__ void __StateVectorAddDifference__(double *SV, double *dx, size_t N) {
    const unsigned threadId = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned offset = threadId * ELEMENTS_PER_THREAD;
    const unsigned upper_limit = offset + ELEMENTS_PER_THREAD;

    if (upper_limit < N) {

#pragma unroll
        for (int T = 0; T < ELEMENTS_PER_THREAD; ++T) {
            const unsigned IDX = offset + T;
            SV[2 * IDX + 0] += dx[2 * IDX + 0];
            SV[2 * IDX + 1] += dx[2 * IDX + 1];
        }

    } else {
        const unsigned REMAINDER = N - offset;

        for (int T = 0; T < REMAINDER; ++T) {
            const unsigned IDX = offset + T;
            if (IDX < N) {
                SV[2 * IDX + 0] += dx[2 * IDX + 0];
                SV[2 * IDX + 1] += dx[2 * IDX + 1];
            }
        }
    }
}

/// =======================================================================================
/// <summary>
/// Accumulate difference from newton-raphson method;  SV[] = SV[] + dx;
/// </summary>
/// <param name="stream"></param>
/// <param name="SV"></param>
/// <param name="dx"></param>
/// <param name="N"></param>
/// <returns></returns>
KERNEL_EXECUTOR void StateVectorAddDifference(cudaStream_t stream, double *SV, double *dx, size_t N) {

    typedef KernelTraits<ELEMENTS_PER_THREAD, DEFAULT_BLOCK_DIM> PointKernelTraits_t;
    const PointKernelTraits_t PointKernelTraits(N);

    const unsigned GRID_DIM = PointKernelTraits.GRID_DIM;
    const unsigned BLOCK_DIM = PointKernelTraits.BLOCK_DIM;
    const unsigned NS = 0;

    __StateVectorAddDifference__<<<GRID_DIM, BLOCK_DIM, NS, stream>>>(SV, dx, N);
}

/// ==================================== STIFFNESS MATRIX =================================

///
/// Free Point ============================================================================
///

template <typename Layout> __device__ void setStiffnessMatrix_FreePoint(int rc, graph::Tensor<Layout> &mt);
///
/// Line ==================================================================================
///

template <typename Layout> __device__ void setStiffnessMatrix_Line(int rc, graph::Tensor<Layout> &mt);

///
/// FixLine         \\\\\\  [empty geometric]
///

template <typename Layout> __device__ void setStiffnessMatrix_FixLine(int rc, graph::Tensor<Layout> &mt);

///
/// Circle ================================================================================
///

template <typename Layout> __device__ void setStiffnessMatrix_Circle(int rc, graph::Tensor<Layout> &mt);

///
/// Arcus ================================================================================
///

template <typename Layout> __device__ void setStiffnessMatrix_Arc(int rc, graph::Tensor<Layout> &mt);

/**
 * @brief Compute Stiffness Matrix on each geometric object.
 *
 * Single cuda thread is responsible for evalution of an assigned geometric object.
 *
 *
 * @param ec
 * @return __global__
 */
template <typename Layout>
__device__ void ComputeStiffnessMatrix_Impl(int tID, ComputationState *ecdata, graph::Tensor<Layout> &tensor,
                                            size_t N) {
    ComputationState *ec = static_cast<ComputationState *>(ecdata);
    if (tID < N) {
        const size_t rc = ec->accGeometricSize[tID]; /// row-column row
        const graph::Geometric *geometric = ec->getGeometricObject(tID);
        switch (geometric->geometricTypeId) {
        case GEOMETRIC_TYPE_ID_FREE_POINT:
            setStiffnessMatrix_FreePoint(rc, tensor);
            break;
        case GEOMETRIC_TYPE_ID_LINE:
            setStiffnessMatrix_Line(rc, tensor);
            break;
        case GEOMETRIC_TYPE_ID_FIX_LINE:
            setStiffnessMatrix_FixLine(rc, tensor);
            break;
        case GEOMETRIC_TYPE_ID_CIRCLE:
            setStiffnessMatrix_Circle(rc, tensor);
            break;
        case GEOMETRIC_TYPE_ID_ARC:
            setStiffnessMatrix_Arc(rc, tensor);
            break;
        default:
            log_error("[gpu/error] geometric type unkown !\n");
            break;
        }
    }
    return;
}

/// =======================================================================================

__global__ void __ComputeStiffnessMatrix__(ComputationState *ecdata, size_t N) {

    /// From Kernel Reference Addressing
    int const tID = blockDim.x * blockIdx.x + threadIdx.x;

    ComputationState *ec = static_cast<ComputationState *>(ecdata);
    /// actually max single block with 1024 threads

    ComputationMode computationMode = ec->computationMode;

#ifdef TENSOR_SPARSE_LAYOUT
    int *INV_P = ec->INV_P;
    int const baseOffset = 0;                               /// into coo vector
    int *const accWriteOffset = ec->accCooWriteStiffTensor; /// relative offset in coo
    int *const cooRowInd = ec->cooColInd;
    int *const cooColInd = ec->cooColInd;
    double *const cooVal = ec->cooVal;
    const bool intention = true;
#endif

    /// RUNTIME DISPATCH

    if (computationMode == ComputationMode::DENSE_LAYOUT) {
        graph::DenseLayout denseLayout = graph::DenseLayout(ec->dimension, 0, 0, ec->A);
        graph::Tensor<graph::DenseLayout> tensor = graph::tensorDevMem(denseLayout, 0, 0);
        ///
        ComputeStiffnessMatrix_Impl(tID, ecdata, tensor, N);
        ///
        return;
    }

#ifdef TENSOR_SPARSE_LAYOUT
    if (computationMode == ComputationMode::SPARSE_LAYOUT) {
        graph::SparseLayout sparseLayout =
            graph::SparseLayout(baseOffset, accWriteOffset, cooRowInd, cooColInd, cooVal);
        graph::Tensor<graph::SparseLayout> tensorSparseLayout = graph::tensorDevMem(sparseLayout, intention);
        ///
        ComputeStiffnessMatrix_Impl(tID, ecdata, tensorSparseLayout, N);
        ///
        return;
    }

    if (computationMode == ComputationMode::DIRECT_LAYOUT) {
        graph::DirectSparseLayout directLayout =
            graph::DirectSparseLayout(baseOffset, accWriteOffset, INV_P, cooRowInd, cooColInd, cooVal);
        graph::Tensor<graph::DirectSparseLayout> tensorDirect = graph::tensorDevMem(directLayout, intention);
        ///
        ComputeStiffnessMatrix_Impl(tID, ecdata, tensorDirect, N);
        ///
        return;
    }
#endif

    log_error("[gpu/tensor] computation mode unknown \n");
}

/// =======================================================================================
/// <summary>
/// Compute Stiffness Matrix on each geometric object.
/// Single cuda thread is responsible for evalution of an assigned geometric object.
/// </summary>
/// <param name="stream"></param>
/// <param name="ecdata"></param>
/// <param name="N"></param>
/// <returns></returns>
KERNEL_EXECUTOR void ComputeStiffnessMatrix(cudaStream_t stream, ComputationState *ecdata, size_t N) {

    typedef KernelTraits<OBJECTS_PER_THREAD, DEFAULT_BLOCK_DIM> GeometricKernelTraits_t;
    const GeometricKernelTraits_t GeometricKernelTraits(N);

    const unsigned GRID_DIM = GeometricKernelTraits.GRID_DIM;
    const unsigned BLOCK_DIM = GeometricKernelTraits.BLOCK_DIM;

    __ComputeStiffnessMatrix__<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(ecdata, N);
}

/// -1 * b from equation reduction
__device__ __constant__ constexpr double SPRING_LOW = CONSTS_SPRING_STIFFNESS_LOW;

__device__ __constant__ constexpr double SPRING_HIGH = CONSTS_SPRING_STIFFNESS_HIGH;

__device__ __constant__ constexpr double SPRING_ALFA = CIRCLE_SPRING_ALFA;

///
/// Free Point ============================================================================
///

template <typename Layout> __device__ void setStiffnessMatrix_FreePoint(int rc, graph::Tensor<Layout> &mt) {
    /**
     * k= I*k
     * [ -ks    ks     0;
     *    ks  -2ks   ks ;
     *     0    ks   -ks];

     */
    // K -mala sztywnosci
    graph::TensorBlock Ks = graph::SmallTensor::diagonal(SPRING_LOW);
    graph::TensorBlock Km = Ks.multiplyC(-1);

    mt.plusSubTensor(rc + 0, rc + 0, Km);
    mt.plusSubTensor(rc + 0, rc + 2, Ks);

    mt.plusSubTensor(rc + 2, rc + 0, Ks);
    mt.plusSubTensor(rc + 2, rc + 2, Km.multiplyC(2.0));
    mt.plusSubTensor(rc + 2, rc + 4, Ks);

    mt.plusSubTensor(rc + 4, rc + 2, Ks);
    mt.plusSubTensor(rc + 4, rc + 4, Km);
}

///
/// Line ==================================================================================
///

template <typename Layout> __device__ void setStiffnessMatrix_Line(int rc, graph::Tensor<Layout> &mt) {
    /**
     * k= I*k
     * [ -ks    ks      0  	  0;
     *    ks  -ks-kb   kb  	  0;
     *     0    kb   -ks-kb   ks;
     *     0  	 0     ks    -ks];
     */
    // K -mala sztywnosci
    graph::TensorBlock Ks = graph::SmallTensor::diagonal(SPRING_LOW);
    // K - duza szytwnosci
    graph::TensorBlock Kb = graph::SmallTensor::diagonal(SPRING_HIGH);
    // -Ks-Kb
    graph::TensorBlock Ksb = Ks.multiplyC(-1).plus(Kb.multiplyC(-1));

    // wiersz pierwszy
    mt.plusSubTensor(rc + 0, rc + 0, Ks.multiplyC(-1));
    mt.plusSubTensor(rc + 0, rc + 2, Ks);
    mt.plusSubTensor(rc + 2, rc + 0, Ks);
    mt.plusSubTensor(rc + 2, rc + 2, Ksb);
    mt.plusSubTensor(rc + 2, rc + 4, Kb);
    mt.plusSubTensor(rc + 4, rc + 2, Kb);
    mt.plusSubTensor(rc + 4, rc + 4, Ksb);
    mt.plusSubTensor(rc + 4, rc + 6, Ks);
    mt.plusSubTensor(rc + 6, rc + 4, Ks);
    mt.plusSubTensor(rc + 6, rc + 6, Ks.multiplyC(-1));
}

///
/// FixLine         \\\\\\  [empty geometric]
///

template <typename Layout> __device__ void setStiffnessMatrix_FixLine(int rc, graph::Tensor<Layout> &mt) {}

///
/// Circle ================================================================================
///

template <typename Layout> __device__ void setStiffnessMatrix_Circle(int rc, graph::Tensor<Layout> &mt) {
    /**
     * k= I*k
     * [ -ks    ks      0  	  0;
     *    ks  -ks-kb   kb  	  0;
     *     0    kb   -ks-kb   ks;
     *     0  	 0     ks    -ks];
     */
    // K -mala sztywnosci
    graph::TensorBlock Ks = graph::SmallTensor::diagonal(SPRING_LOW);
    // K - duza szytwnosci
    graph::TensorBlock Kb = graph::SmallTensor::diagonal(SPRING_HIGH * CIRCLE_SPRING_ALFA);
    // -Ks-Kb
    graph::TensorBlock Ksb = Ks.multiplyC(-1).plus(Kb.multiplyC(-1));

    // wiersz pierwszy
    mt.plusSubTensor(rc + 0, rc + 0, Ks.multiplyC(-1));
    mt.plusSubTensor(rc + 0, rc + 2, Ks);
    mt.plusSubTensor(rc + 2, rc + 0, Ks);
    mt.plusSubTensor(rc + 2, rc + 2, Ksb);
    mt.plusSubTensor(rc + 2, rc + 4, Kb);
    mt.plusSubTensor(rc + 4, rc + 2, Kb);
    mt.plusSubTensor(rc + 4, rc + 4, Ksb);
    mt.plusSubTensor(rc + 4, rc + 6, Ks);
    mt.plusSubTensor(rc + 6, rc + 4, Ks);
    mt.plusSubTensor(rc + 6, rc + 6, Ks.multiplyC(-1));
}

///
/// Arcus ================================================================================
///

template <typename Layout> __device__ void setStiffnessMatrix_Arc(int rc, graph::Tensor<Layout> &mt) {
    // K -mala sztywnosci
    graph::TensorBlock Kb = graph::SmallTensor::diagonal(SPRING_HIGH);
    graph::TensorBlock Ks = graph::SmallTensor::diagonal(SPRING_LOW);

    graph::TensorBlock mKs = Ks.multiplyC(-1);
    graph::TensorBlock mKb = Kb.multiplyC(-1);
    graph::TensorBlock KsKbm = mKs.plus(mKb);

    mt.plusSubTensor(rc + 0, rc + 0, mKs);
    mt.plusSubTensor(rc + 0, rc + 8, Ks); // a

    mt.plusSubTensor(rc + 2, rc + 2, mKs);
    mt.plusSubTensor(rc + 2, rc + 8, Ks); // b

    mt.plusSubTensor(rc + 4, rc + 4, mKs);
    mt.plusSubTensor(rc + 4, rc + 10, Ks); // c

    mt.plusSubTensor(rc + 6, rc + 6, mKs);
    mt.plusSubTensor(rc + 6, rc + 12, Ks); // d

    mt.plusSubTensor(rc + 8, rc + 0, Ks);
    mt.plusSubTensor(rc + 8, rc + 2, Ks);
    mt.plusSubTensor(rc + 8, rc + 8, KsKbm.multiplyC(2.0));
    mt.plusSubTensor(rc + 8, rc + 10, Kb);
    mt.plusSubTensor(rc + 8, rc + 12, Kb); // p1

    mt.plusSubTensor(rc + 10, rc + 4, Ks);
    mt.plusSubTensor(rc + 10, rc + 8, Kb);
    mt.plusSubTensor(rc + 10, rc + 10, KsKbm); // p2

    mt.plusSubTensor(rc + 12, rc + 6, Ks);
    mt.plusSubTensor(rc + 12, rc + 8, Kb);
    mt.plusSubTensor(rc + 12, rc + 12, KsKbm); // p3
}

///
/// ================================ FORCE INTENSITY ==================== ///
///

///
/// Free Point ============================================================================
///

template <typename Layout>
__device__ void setForceIntensity_FreePoint(int row, graph::Geometric const *geometric, ComputationState *ec,
                                            graph::Tensor<Layout> &mt);

///
/// Line    ===============================================================================
///

template <typename Layout>
__device__ void setForceIntensity_Line(int row, graph::Geometric const *geometric, ComputationState *ec,
                                       graph::Tensor<Layout> &mt);

///
/// FixLine ===============================================================================
///

template <typename Layout>
__device__ void setForceIntensity_FixLine(int row, graph::Geometric const *geometric, ComputationState *ec,
                                          graph::Tensor<Layout> &mt);

///
/// Circle  ===============================================================================
///

template <typename Layout>
__device__ void setForceIntensity_Circle(int row, graph::Geometric const *geometric, ComputationState *ec,
                                         graph::Tensor<Layout> &mt);

///
/// Arc ===================================================================================
///

template <typename Layout>
__device__ void setForceIntensity_Arc(int row, graph::Geometric const *geometric, ComputationState *ec,
                                      graph::Tensor<Layout> &mt);

///
/// Evaluate Force Intensity ==============================================================
///
__global__ void EvaluateForceIntensity_Impl(ComputationState *ecdata, size_t N) {

    /// Kernel Reference Addressing
    int tId = blockDim.x * blockIdx.x + threadIdx.x;

    ComputationState *ec = (ComputationState *)ecdata;

    /// unpack tensor for evaluation

    const bool intention = true; // vector operations
    graph::DenseLayout layout = graph::DenseLayout(ec->dimension, 0, 0, ec->b);
    graph::Tensor<graph::DenseLayout> mt = graph::tensorDevMem(layout, 0, 0, intention);

    if (tId < N) {
        const graph::Geometric *geometric = ec->getGeometricObject(tId);

        const size_t row = ec->accGeometricSize[tId];

        switch (geometric->geometricTypeId) {

        case GEOMETRIC_TYPE_ID_FREE_POINT:
            setForceIntensity_FreePoint(row, geometric, ec, mt);
            break;

        case GEOMETRIC_TYPE_ID_LINE:
            setForceIntensity_Line(row, geometric, ec, mt);
            break;

        case GEOMETRIC_TYPE_ID_FIX_LINE:
            setForceIntensity_FixLine(row, geometric, ec, mt);
            break;

        case GEOMETRIC_TYPE_ID_CIRCLE:
            setForceIntensity_Circle(row, geometric, ec, mt);
            break;

        case GEOMETRIC_TYPE_ID_ARC:
            setForceIntensity_Arc(row, geometric, ec, mt);
            break;
        default:
            break;
        }
    }

    return;
}

KERNEL_EXECUTOR void EvaluateForceIntensity(cudaStream_t stream, ComputationState *ecdata, size_t N) {

    typedef KernelTraits<OBJECTS_PER_THREAD, DEFAULT_BLOCK_DIM> GeometricKernelTraits_t;
    const GeometricKernelTraits_t GeometricKernelTraits(N);

    const unsigned GRID_DIM = GeometricKernelTraits.GRID_DIM;
    const unsigned BLOCK_DIM = GeometricKernelTraits.BLOCK_DIM;
    const unsigned NS = 0;

    EvaluateForceIntensity_Impl<<<GRID_DIM, BLOCK_DIM, NS, stream>>>(ecdata, N);
}

///
/// Free Point ============================================================================
///

template <typename Layout>
__device__ void setForceIntensity_FreePoint(int row, graph::Geometric const *geometric, ComputationState *ec,
                                            graph::Tensor<Layout> &mt) {
    graph::Vector const &a = ec->getPoint(geometric->a);
    graph::Vector const &b = ec->getPoint(geometric->b);
    graph::Vector const &p1 = ec->getPoint(geometric->p1);

    // declaration time const
    graph::Point const &p1c = ec->getPointRef(geometric->p1);

    const double d_a_p1 = abs(p1c.minus(a).length()) * 0.1;
    const double d_p1_b = abs(b.minus(p1c).length()) * 0.1;

    // 8 = 4*2 (4 punkty kontrolne)

    // F12 - sily w sprezynach
    graph::Vector f12 = p1.minus(a).unit().product(-SPRING_LOW).product(p1.minus(a).length() - d_a_p1);
    // F23
    graph::Vector f23 = b.minus(p1).unit().product(-SPRING_LOW).product(b.minus(p1).length() - d_p1_b);

    // F1 - sily na poszczegolne punkty
    mt.setVector(row + 0, 0, f12);
    // F2
    mt.setVector(row + 2, 0, f23.minus(f12));
    // F3
    mt.setVector(row + 4, 0, f23.product(-1));
}

///
/// Line    ===============================================================================
///

template <typename Layout>
__device__ void setForceIntensity_Line(int row, graph::Geometric const *geometric, ComputationState *ec,
                                       graph::Tensor<Layout> &mt) {
    graph::Vector const &a = ec->getPoint(geometric->a);
    graph::Vector const &b = ec->getPoint(geometric->b);
    graph::Vector const &p1 = ec->getPoint(geometric->p1);
    graph::Vector const &p2 = ec->getPoint(geometric->p2);

    // declaration time const
    graph::Point const &p1c = ec->getPointRef(geometric->p1);
    graph::Point const &p2c = ec->getPointRef(geometric->p2);

    const double d_a_p1 = abs(p1c.minus(a).length());
    const double d_p1_p2 = abs(p2c.minus(p1c).length());
    const double d_p2_b = abs(b.minus(p2c).length());

    // 8 = 4*2 (4 punkty kontrolne)
    //
    // F12 - sily w sprezynach
    graph::Vector f12 = p1.minus(a).unit().product(-SPRING_LOW).product(p1.minus(a).length() - d_a_p1);
    // F23
    graph::Vector f23 = p2.minus(p1).unit().product(-SPRING_HIGH).product(p2.minus(p1).length() - d_p1_p2);
    // F34
    graph::Vector f34 = b.minus(p2).unit().product(-SPRING_LOW).product(b.minus(p2).length() - d_p2_b);

    // F1 - silu na poszczegolne punkty
    mt.setVector(row + 0, 0, f12);
    // F2
    mt.setVector(row + 2, 0, f23.minus(f12));
    // F3
    mt.setVector(row + 4, 0, f34.minus(f23));
    // F4
    mt.setVector(row + 6, 0, f34.product(-1.0));
}

///
/// FixLine ===============================================================================
///

template <typename Layout>
__device__ void setForceIntensity_FixLine(int row, graph::Geometric const *geometric, ComputationState *ec,
                                          graph::Tensor<Layout> &mt) {}

///
/// Circle  ===============================================================================
///

template <typename Layout>
__device__ void setForceIntensity_Circle(int row, graph::Geometric const *geometric, ComputationState *ec,
                                         graph::Tensor<Layout> &mt) {
    graph::Vector const &a = ec->getPoint(geometric->a);
    graph::Vector const &b = ec->getPoint(geometric->b);
    graph::Vector const &p1 = ec->getPoint(geometric->p1);
    graph::Vector const &p2 = ec->getPoint(geometric->p2);

    // declaration time const
    graph::Point const &p1c = ec->getPointRef(geometric->p1);
    graph::Point const &p2c = ec->getPointRef(geometric->p2);

    const double d_a_p1 = abs(p1c.minus(a).length());
    const double d_p1_p2 = abs(p2c.minus(p1c).length());
    const double d_p2_b = abs(b.minus(p2c).length());

    // 8 = 4*2 (4 punkty kontrolne)

    // F12 - sily w sprezynach
    graph::Vector f12 = p1.minus(a).unit().product(-SPRING_LOW).product(p1.minus(a).length() - d_a_p1);
    // F23
    graph::Vector f23 =
        p2.minus(p1).unit().product(-SPRING_HIGH * SPRING_ALFA).product(p2.minus(p1).length() - d_p1_p2);
    // F34
    graph::Vector f34 = b.minus(p2).unit().product(-SPRING_LOW).product(b.minus(p2).length() - d_p2_b);

    // F1 - silu na poszczegolne punkty
    mt.setVector(row + 0, 0, f12);
    // F2
    mt.setVector(row + 2, 0, f23.minus(f12));
    // F3
    mt.setVector(row + 4, 0, f34.minus(f23));
    // F4
    mt.setVector(row + 6, 0, f34.product(-1.0));
}

///
/// Arc ===================================================================================
///

template <typename Layout>
__device__ void setForceIntensity_Arc(int row, graph::Geometric const *geometric, ComputationState *ec,
                                      graph::Tensor<Layout> &mt) {
    graph::Vector const &a = ec->getPoint(geometric->a);
    graph::Vector const &b = ec->getPoint(geometric->b);
    graph::Vector const &c = ec->getPoint(geometric->c);
    graph::Vector const &d = ec->getPoint(geometric->d);
    graph::Vector const &p1 = ec->getPoint(geometric->p1);
    graph::Vector const &p2 = ec->getPoint(geometric->p2);
    graph::Vector const &p3 = ec->getPoint(geometric->p3);

    // declaration time const
    graph::Point const &p1c = ec->getPointRef(geometric->p1);
    graph::Point const &p2c = ec->getPointRef(geometric->p2);
    graph::Point const &p3c = ec->getPointRef(geometric->p3);

    /// naciag wstepny lepiej sie zbiegaja
    double d_a_p1 = abs(p1c.minus(a).length());
    double d_b_p1 = abs(p1c.minus(b).length());
    double d_p1_p2 = abs(p2c.minus(p1c).length());
    double d_p1_p3 = abs(p3c.minus(p1c).length());
    double d_p3_d = abs(d.minus(p3c).length());
    double d_p2_c = abs(c.minus(p2c).length());

    graph::Vector fap1 = p1.minus(a).unit().product(-SPRING_LOW).product(p1.minus(a).length() - d_a_p1);
    graph::Vector fbp1 = p1.minus(b).unit().product(-SPRING_LOW).product(p1.minus(b).length() - d_b_p1);
    graph::Vector fp1p2 = p2.minus(p1).unit().product(-SPRING_HIGH).product(p2.minus(p1).length() - d_p1_p2);
    graph::Vector fp1p3 = p3.minus(p1).unit().product(-SPRING_HIGH).product(p3.minus(p1).length() - d_p1_p3);
    graph::Vector fp2c = c.minus(p2).unit().product(-SPRING_LOW).product(c.minus(p2).length() - d_p2_c);
    graph::Vector fp3d = d.minus(p3).unit().product(-SPRING_LOW).product(d.minus(p3).length() - d_p3_d);

    mt.setVector(row + 0, 0, fap1);
    mt.setVector(row + 2, 0, fbp1);
    mt.setVector(row + 4, 0, fp2c.product(-1));
    mt.setVector(row + 6, 0, fp3d.product(-1));
    mt.setVector(row + 8, 0, fp1p2.plus(fp1p3).minus(fap1).minus(fbp1));
    mt.setVector(row + 10, 0, fp2c.minus(fp1p2));
    mt.setVector(row + 12, 0, fp3d.minus(fp1p3));
}

/// ==================================== CONSTRAINT VALUE =================================

///
/// ConstraintFixPoint ====================================================================
///
template <typename Layout>
__device__ void setValueConstraintFixPoint(int row, graph::Constraint const *constraint, ComputationState *ec,
                                           graph::Tensor<Layout> &mt);

///
/// ConstraintParametrizedXfix ============================================================
///
template <typename Layout>
__device__ void setValueConstraintParametrizedXfix(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor<Layout> &mt);

///
/// ConstraintParametrizedYfix ============================================================
///
template <typename Layout>
__device__ void setValueConstraintParametrizedYfix(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor<Layout> &mt);

///
/// ConstraintConnect2Points ==============================================================
///
template <typename Layout>
__device__ void setValueConstraintConnect2Points(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                 graph::Tensor<Layout> &mt);

///
/// ConstraintHorizontalPoint =============================================================
///
template <typename Layout>
__device__ void setValueConstraintHorizontalPoint(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                  graph::Tensor<Layout> &mt);

///
/// ConstraintVerticalPoint ===============================================================
///
template <typename Layout>
__device__ void setValueConstraintVerticalPoint(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                graph::Tensor<Layout> &mt);

///
/// ConstraintLinesParallelism ============================================================
///
template <typename Layout>
__device__ void setValueConstraintLinesParallelism(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor<Layout> &mt);

///
/// ConstraintLinesPerpendicular ==========================================================
///
template <typename Layout>
__device__ void setValueConstraintLinesPerpendicular(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                     graph::Tensor<Layout> &mt);

///
/// ConstraintEqualLength =================================================================
///
template <typename Layout>
__device__ void setValueConstraintEqualLength(int row, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor<Layout> &mt);

///
/// ConstraintParametrizedLength ==========================================================
///
template <typename Layout>
__device__ void setValueConstraintParametrizedLength(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                     graph::Tensor<Layout> &mt);

///
/// ConstrainTangency =====================================================================
///
template <typename Layout>
__device__ void setValueConstraintTangency(int row, graph::Constraint const *constraint, ComputationState *ec,
                                           graph::Tensor<Layout> &mt);

///
/// ConstraintCircleTangency ==============================================================
///
template <typename Layout>
__device__ void setValueConstraintCircleTangency(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                 graph::Tensor<Layout> &mt);

///
/// ConstraintDistance2Points =============================================================
///
template <typename Layout>
__device__ void setValueConstraintDistance2Points(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                  graph::Tensor<Layout> &mt);

///
/// ConstraintDistancePointLine ===========================================================
///
template <typename Layout>
__device__ void setValueConstraintDistancePointLine(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                    graph::Tensor<Layout> &mt);

///
/// ConstraintAngle2Lines =================================================================
///
template <typename Layout>
__device__ void setValueConstraintAngle2Lines(int row, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor<Layout> &mt);

///
/// ConstraintSetHorizontal ===============================================================
///
template <typename Layout>
__device__ void setValueConstraintSetHorizontal(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                graph::Tensor<Layout> &mt);

///
/// ConstraintSetVertical =================================================================
///
template <typename Layout>
__device__ void setValueConstraintSetVertical(int row, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor<Layout> &mt);

///
/// Evaluate Constraint Value =============================================================
///
__global__ void __EvaluateConstraintValue__(ComputationState *ecdata, size_t N) {

    /// From Kernel Reference Addressing
    int tId = blockDim.x * blockIdx.x + threadIdx.x;

    ComputationState *ec = static_cast<ComputationState *>(ecdata);

    const bool intention = true;
    const graph::DenseLayout layout = graph::DenseLayout(ec->dimension, ec->size, 0, ec->b);
    graph::Tensor<graph::DenseLayout> mt = graph::tensorDevMem(layout, 0, 0, intention);

    if (tId < N) {
        const graph::Constraint *constraint = ec->getConstraint(tId);

        const int row = ec->accConstraintSize[tId];

        switch (constraint->constraintTypeId) {
        case CONSTRAINT_TYPE_ID_FIX_POINT:
            setValueConstraintFixPoint(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_XFIX:
            setValueConstraintParametrizedXfix(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_YFIX:
            setValueConstraintParametrizedYfix(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_CONNECT_2_POINTS:
            setValueConstraintConnect2Points(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_HORIZONTAL_POINT:
            setValueConstraintHorizontalPoint(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_VERTICAL_POINT:
            setValueConstraintVerticalPoint(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PARALLELISM:
            setValueConstraintLinesParallelism(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR:
            setValueConstraintLinesPerpendicular(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_EQUAL_LENGTH:
            setValueConstraintEqualLength(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_LENGTH:
            setValueConstraintParametrizedLength(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_TANGENCY:
            setValueConstraintTangency(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_CIRCLE_TANGENCY:
            setValueConstraintCircleTangency(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_2_POINTS:
            setValueConstraintDistance2Points(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_POINT_LINE:
            setValueConstraintDistancePointLine(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_ANGLE_2_LINES:
            setValueConstraintAngle2Lines(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_SET_HORIZONTAL:
            setValueConstraintSetHorizontal(row, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_SET_VERTICAL:
            setValueConstraintSetVertical(row, constraint, ec, mt);
            break;
        default:
            break;
        }
        return;
    }
}

/// =======================================================================================
/// <summary>
/// Evaluate constraint values for right-hand-side vector.
/// </summary>
/// <param name="stream"></param>
/// <param name="ecdata"></param>
/// <param name="N"></param>
/// <returns></returns>
KERNEL_EXECUTOR void EvaluateConstraintValue(cudaStream_t stream, ComputationState *ecdata, size_t N) {

    typedef KernelTraits<OBJECTS_PER_THREAD, DEFAULT_BLOCK_DIM> ConstraintKernelTraits_t;
    const ConstraintKernelTraits_t ConstraintKernelTraits(N);

    const unsigned GRID_DIM = ConstraintKernelTraits.GRID_DIM;
    const unsigned BLOCK_DIM = ConstraintKernelTraits.BLOCK_DIM;
    const unsigned NS = 0;

    __EvaluateConstraintValue__<<<GRID_DIM, BLOCK_DIM, NS, stream>>>(ecdata, N);
}

///
/// ConstraintFixPoint ====================================================================
///
template <typename Layout>
__device__ void setValueConstraintFixPoint(int row, graph::Constraint const *constraint, ComputationState *ec,
                                           graph::Tensor<Layout> &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector ko_vec = graph::Vector(constraint->vecX, constraint->vecY);

    ///
    graph::Vector value = k.minus(ko_vec).product(-1);

    ///
    mt.setVector(row, 0, value);
}

///
/// ConstraintParametrizedXfix ============================================================
///
template <typename Layout>
__device__ void setValueConstraintParametrizedXfix(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor<Layout> &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);

    ///
    double value = -1 * (k.x - param->value);
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintParametrizedYfix ============================================================
///
template <typename Layout>
__device__ void setValueConstraintParametrizedYfix(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor<Layout> &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);

    ///
    double value = -1 * (k.y - param->value);
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintConnect2Points ==============================================================
///
template <typename Layout>
__device__ void setValueConstraintConnect2Points(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                 graph::Tensor<Layout> &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);

    ///
    graph::Vector value = k.minus(l).product(-1);
    ///
    mt.setVector(row, 0, value);
}

///
/// ConstraintHorizontalPoint =============================================================
///
template <typename Layout>
__device__ void setValueConstraintHorizontalPoint(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                  graph::Tensor<Layout> &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);

    ///
    double value = -1 * (k.x - l.x);
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintVerticalPoint ===============================================================
///
template <typename Layout>
__device__ void setValueConstraintVerticalPoint(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                graph::Tensor<Layout> &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);

    ///
    double value = -1 * (k.y - l.y);
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintLinesParallelism ============================================================
///
template <typename Layout>
__device__ void setValueConstraintLinesParallelism(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor<Layout> &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);

    ///
    double value = -1 * LK.cross(NM);
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintLinesPerpendicular ==========================================================
///
template <typename Layout>
__device__ void setValueConstraintLinesPerpendicular(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                     graph::Tensor<Layout> &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector KL = k.minus(l);
    const graph::Vector MN = m.minus(n);

    ///
    double value = -1 * KL.product(MN);
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintEqualLength =================================================================
///
template <typename Layout>
__device__ void setValueConstraintEqualLength(int row, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor<Layout> &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);

    ///
    double value = -1 * (LK.length() - NM.length());
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintParametrizedLength ==========================================================
///
template <typename Layout>
__device__ void setValueConstraintParametrizedLength(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                     graph::Tensor<Layout> &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);

    ///
    double value = -1 * (param->value * LK.length() - NM.length());
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstrainTangency =====================================================================
///
template <typename Layout>
__device__ void setValueConstraintTangency(int row, graph::Constraint const *constraint, ComputationState *ec,
                                           graph::Tensor<Layout> &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);
    const graph::Vector MK = m.minus(k);

    ///
    double value = -1 * (LK.cross(MK) * LK.cross(MK) - LK.product(LK) * NM.product(NM));
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintCircleTangency ==============================================================
///
template <typename Layout>
__device__ void setValueConstraintCircleTangency(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                 graph::Tensor<Layout> &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);
    const graph::Vector MK = m.minus(k);

    ///
    double value = -1 * (LK.length() + NM.length() - MK.length());
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintDistance2Points =============================================================
///
template <typename Layout>
__device__ void setValueConstraintDistance2Points(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                  graph::Tensor<Layout> &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);

    ///
    double value = -1 * (l.minus(k).length() - param->value);
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintDistancePointLine ===========================================================
///
template <typename Layout>
__device__ void setValueConstraintDistancePointLine(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                    graph::Tensor<Layout> &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);

    const graph::Vector LK = l.minus(k);
    const graph::Vector MK = m.minus(k);

    ///
    double value = -1 * (LK.cross(MK) - param->value * LK.length());
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintAngle2Lines =================================================================
///
template <typename Layout>
__device__ void setValueConstraintAngle2Lines(int row, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor<Layout> &mt) {
    /// coordinate system o_x axis
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);

    ///
    double value = -1 * (LK.product(NM) - LK.length() * NM.length() * cos(toRadians(param->value)));
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintSetHorizontal ===============================================================
///
template <typename Layout>
__device__ void setValueConstraintSetHorizontal(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                graph::Tensor<Layout> &mt) {
    /// coordinate system oY axis
    const graph::Vector m = graph::Vector(0.0, 0.0);
    const graph::Vector n = graph::Vector(0.0, 100.0);

    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);

    ///
    double value = -1 * k.minus(l).product(m.minus(n));
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintSetVertical =================================================================
///
template <typename Layout>
__device__ void setValueConstraintSetVertical(int row, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor<Layout> &mt) {
    /// coordinate system oX axis
    const graph::Vector m = graph::Vector(0.0, 0.0);
    const graph::Vector n = graph::Vector(100.0, 0.0);
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);

    ///
    double value = -1 * k.minus(l).product(m.minus(n));
    ///
    mt.setValue(row, 0, value);
}

/// ============================ CONSTRAINT JACOBIAN MATRIX  ==================================

///
/// ConstraintFixPoint    =================================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintFixPoint(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                              Tensor &mt);

///
/// ConstraintParametrizedXfix    =========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintParametrizedXfix(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, Tensor &mt);
///
/// ConstraintParametrizedYfix    =========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintParametrizedYfix(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, Tensor &mt);

///
/// ConstraintConnect2Points    ===========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintConnect2Points(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                    Tensor &mt);

///
/// ConstraintHorizontalPoint    ==========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintHorizontalPoint(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                     Tensor &mt);

///
/// ConstraintVerticalPoint    ============================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintVerticalPoint(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                   Tensor &mt);

///
/// ConstraintLinesParallelism    =========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintLinesParallelism(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, Tensor &mt);
///
/// ConstraintLinesPerpendicular    =======================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintLinesPerpendicular(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, Tensor &mt);

///
/// ConstraintEqualLength    ==============================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintEqualLength(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                 Tensor &mt);

///
/// ConstraintParametrizedLength    =======================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintParametrizedLength(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, Tensor &mt);

///
/// ConstrainTangency    ==================================================================
///
template <typename Tensor>
__device__ void setJacobianConstrainTangency(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                             Tensor &mt);

///
/// ConstraintCircleTangency    ===========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintCircleTangency(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                    Tensor &mt);

///
/// ConstraintDistance2Points    ==========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintDistance2Points(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                     Tensor &mt);

///
/// ConstraintDistancePointLine    ========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintDistancePointLine(int tID, graph::Constraint const *constraint,
                                                       ComputationState *ec, Tensor &mt);

///
/// ConstraintAngle2Lines    ==============================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintAngle2Lines(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                 Tensor &mt);

///
/// ConstraintSetHorizontal    ============================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintSetHorizontal(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                   Tensor &mt);

///
/// ConstraintSetVertical    ==============================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintSetVertical(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                 Tensor &mt);

/// <summary>
/// Evaluate Constraint Jacobian (FI) - (dfi/dq)   lower slice matrix of A
/// </summary>
/// <param name="tId"></param>
/// <param name="ecdata"></param>
/// <param name="mt1"></param>
/// <param name="N"></param>
/// <returns></returns>
template <typename Tensor>
__device__ void EvaluateConstraintJacobian_Impl(int tID, ComputationState *ecdata, Tensor &mt1, size_t N) {
    ComputationState *ec = static_cast<ComputationState *>(ecdata);
    /// unpack tensor for evaluation
    /// COLUMN_ORDER - tensor A
    if (tID < N) {
        const graph::Constraint *constraint = ec->getConstraint(tID);
        switch (constraint->constraintTypeId) {
        case CONSTRAINT_TYPE_ID_FIX_POINT:
            setJacobianConstraintFixPoint(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_XFIX:
            setJacobianConstraintParametrizedXfix(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_YFIX:
            setJacobianConstraintParametrizedYfix(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_CONNECT_2_POINTS:
            setJacobianConstraintConnect2Points(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_HORIZONTAL_POINT:
            setJacobianConstraintHorizontalPoint(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_VERTICAL_POINT:
            setJacobianConstraintVerticalPoint(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PARALLELISM:
            setJacobianConstraintLinesParallelism(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR:
            setJacobianConstraintLinesPerpendicular(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_EQUAL_LENGTH:
            setJacobianConstraintEqualLength(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_LENGTH:
            setJacobianConstraintParametrizedLength(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_TANGENCY:
            setJacobianConstrainTangency(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_CIRCLE_TANGENCY:
            setJacobianConstraintCircleTangency(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_2_POINTS:
            setJacobianConstraintDistance2Points(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_POINT_LINE:
            setJacobianConstraintDistancePointLine(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_ANGLE_2_LINES:
            setJacobianConstraintAngle2Lines(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_SET_HORIZONTAL:
            setJacobianConstraintSetHorizontal(tID, constraint, ec, mt1);
            break;
        case CONSTRAINT_TYPE_ID_SET_VERTICAL:
            setJacobianConstraintSetVertical(tID, constraint, ec, mt1);
            break;
        default:
            break;
        }
        return;
    }
}

/// =======================================================================================
__global__ void __EvaluateConstraintJacobian__(ComputationState *ecdata, size_t N) {

    /// From Kernel Reference Addressing
    int const tId = blockDim.x * blockIdx.x + threadIdx.x;

    ComputationState *ec = static_cast<ComputationState *>(ecdata);

    const ComputationMode computationMode = ec->computationMode;

    const int constraintOffset = ec->accConstraintSize[tId];
#ifdef TENSOR_SPARSE_LAYOUT
    int *const INV_P = ec->INV_P;
    int const baseOffset = ec->cooWritesStiffSize;             /// into coo vector
    int *const accWriteOffset = ec->accCooWriteJacobianTensor; /// relative coo offset
    int *const cooRowInd = ec->cooColInd;
    int *const cooColInd = ec->cooColInd;
    double *const cooVal = ec->cooVal;
#endif
    const bool intention = false;

    /// RUNTIME DISPATCH

    if (computationMode == ComputationMode::DENSE_LAYOUT) {
        graph::DenseLayout denseLayout = graph::DenseLayout(ec->dimension, ec->size, 0, ec->A);
        graph::Tensor<graph::DenseLayout> tensor = graph::tensorDevMem(denseLayout, constraintOffset, 0, intention);
        ///
        EvaluateConstraintJacobian_Impl(tId, ecdata, tensor, N);
        ///
        return;
    }

#ifdef TENSOR_SPARSE_LAYOUT
    if (computationMode == ComputationMode::SPARSE_LAYOUT) {
        graph::SparseLayout sparseLayout =
            graph::SparseLayout(baseOffset, accWriteOffset, cooRowInd, cooColInd, cooVal);
        graph::Tensor<graph::SparseLayout> tensorSparseLayout = graph::tensorDevMem(sparseLayout, intention);
        ///
        EvaluateConstraintJacobian_Impl(tId, ecdata, tensorSparseLayout, N);
        ///
        return;
    }

    if (computationMode == ComputationMode::DIRECT_LAYOUT) {
        graph::DirectSparseLayout directLayout =
            graph::DirectSparseLayout(baseOffset, accWriteOffset, INV_P, cooRowInd, cooColInd, cooVal);
        graph::Tensor<graph::DirectSparseLayout> tensorDirect = graph::tensorDevMem(directLayout, intention);
        ///
        EvaluateConstraintJacobian_Impl(tId, ecdata, tensorDirect, N);
        ///
        return;
    }
#endif

    log_error("[gpu/tensor] computation mode unknown \n");
}

/// =======================================================================================
/// <summary>
/// Evaluate Constraint Jacobian (FI) - (dfi/dq)   lower slice matrix of A
/// </summary>
/// <param name="stream"></param>
/// <param name="ecdata"></param>
/// <param name="N"></param>
/// <returns></returns>
KERNEL_EXECUTOR void EvaluateConstraintJacobian(cudaStream_t stream, ComputationState *ecdata, size_t N) {

    typedef KernelTraits<OBJECTS_PER_THREAD, DEFAULT_BLOCK_DIM> ConstraintKernelTraits_t;
    const ConstraintKernelTraits_t ConstraintKernelTraits(N);

    const unsigned GRID_DIM = ConstraintKernelTraits.GRID_DIM;
    const unsigned BLOCK_DIM = ConstraintKernelTraits.BLOCK_DIM;
    const unsigned NS = 0;

    __EvaluateConstraintJacobian__<<<GRID_DIM, BLOCK_DIM, NS, stream>>>(ecdata, N);
}

/// =======================================================================================
/// <summary>
/// Evaluate Constraint Transposed Jacobian  (FI)' - (dfi/dq)'   tr-transponowane - upper slice matrix  of A
/// </summary>
/// <param name="tId"></param>
/// <param name="ecdata"></param>
/// <param name="mt2"></param>
/// <param name="N"></param>
/// <returns></returns>
template <typename Tensor>
__device__ void EvaluateConstraintTRJacobian_Impl(int tID, ComputationState *ecdata, Tensor &mt2, size_t N) {
    ComputationState *ec = static_cast<ComputationState *>(ecdata);
    /// unpack tensor for evaluation
    /// COLUMN_ORDER - tensor A
    if (tID < N) {
        const graph::Constraint *constraint = ec->getConstraint(tID);
        switch (constraint->constraintTypeId) {
        case CONSTRAINT_TYPE_ID_FIX_POINT:
            setJacobianConstraintFixPoint(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_XFIX:
            setJacobianConstraintParametrizedXfix(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_YFIX:
            setJacobianConstraintParametrizedYfix(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_CONNECT_2_POINTS:
            setJacobianConstraintConnect2Points(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_HORIZONTAL_POINT:
            setJacobianConstraintHorizontalPoint(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_VERTICAL_POINT:
            setJacobianConstraintVerticalPoint(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PARALLELISM:
            setJacobianConstraintLinesParallelism(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR:
            setJacobianConstraintLinesPerpendicular(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_EQUAL_LENGTH:
            setJacobianConstraintEqualLength(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_LENGTH:
            setJacobianConstraintParametrizedLength(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_TANGENCY:
            setJacobianConstrainTangency(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_CIRCLE_TANGENCY:
            setJacobianConstraintCircleTangency(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_2_POINTS:
            setJacobianConstraintDistance2Points(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_POINT_LINE:
            setJacobianConstraintDistancePointLine(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_ANGLE_2_LINES:
            setJacobianConstraintAngle2Lines(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_SET_HORIZONTAL:
            setJacobianConstraintSetHorizontal(tID, constraint, ec, mt2);
            break;
        case CONSTRAINT_TYPE_ID_SET_VERTICAL:
            setJacobianConstraintSetVertical(tID, constraint, ec, mt2);
            break;
        default:
            break;
        }
        return;
    }
}

/// =======================================================================================
/// <summary>
///
/// </summary>
/// <param name="ecdata"></param>
/// <param name="N"></param>
/// <returns></returns>
__global__ void __EvaluateConstraintTRJacobian__(ComputationState *ecdata, size_t N) {

    /// From Kernel Reference Addressing
    int const tId = blockDim.x * blockIdx.x + threadIdx.x;

    ComputationState *ec = static_cast<ComputationState *>(ecdata);

    const ComputationMode computationMode = ec->computationMode;

    const int constraintOffset = (ec->size + ec->accConstraintSize[tId]);

#ifdef TENSOR_SPARSE_LAYOUT
    int *const INV_P = ec->INV_P;
    int const baseOffset = ec->cooWritesStiffSize + ec->cooWirtesJacobianSize; /// into coo vector
    int *const accWriteOffset = ec->accCooWriteJacobianTensor;                 /// relative coo offset
    int *const cooRowInd = ec->cooColInd;
    int *const cooColInd = ec->cooColInd;
    double *const cooVal = ec->cooVal;
#endif
    const bool intention = false;

    /// RUNTIME DISPATCH

    if (computationMode == ComputationMode::DENSE_LAYOUT) {
        graph::DenseLayout denseLayout = graph::DenseLayout(ec->dimension, 0, constraintOffset, ec->A);
        graph::AdapterTensor<graph::DenseLayout> tensor = graph::transposeTensorDevMem(denseLayout, intention);
        ///
        EvaluateConstraintTRJacobian_Impl(tId, ecdata, tensor, N);
        ///
        return;
    }

#ifdef TENSOR_SPARSE_LAYOUT
    if (computationMode == ComputationMode::SPARSE_LAYOUT) {
        graph::SparseLayout sparseLayout =
            graph::SparseLayout(baseOffset, accWriteOffset, cooRowInd, cooColInd, cooVal);
        graph::Tensor<graph::SparseLayout> tensorSparseLayout = graph::tensorDevMem(sparseLayout, intention);
        ///
        EvaluateConstraintTRJacobian_Impl(tId, ecdata, tensorSparseLayout, N);
        ///
        return;
    }

    if (computationMode == ComputationMode::DIRECT_LAYOUT) {
        graph::DirectSparseLayout directLayout =
            graph::DirectSparseLayout(baseOffset, accWriteOffset, INV_P, cooRowInd, cooColInd, cooVal);
        graph::Tensor<graph::DirectSparseLayout> tensorDirect = graph::tensorDevMem(directLayout, intention);
        ///
        EvaluateConstraintTRJacobian_Impl(tId, ecdata, tensorDirect, N);
        ///
        return;
    }
#endif

    log_error("[gpu/tensor] computation mode unknown \n");
}

/// =======================================================================================
/// <summary>
/// Evaluate Constraint Transposed Jacobian  (FI)' - (dfi/dq)'   tr-transponowane - upper slice matrix  of A
/// </summary>
/// <param name="stream"></param>
/// <param name="ecdata"></param>
/// <param name="N"></param>
/// <returns></returns>
KERNEL_EXECUTOR void EvaluateConstraintTRJacobian(cudaStream_t stream, ComputationState *ecdata, size_t N) {

    typedef KernelTraits<OBJECTS_PER_THREAD, DEFAULT_BLOCK_DIM> ConstraintKernelTraits_t;
    const ConstraintKernelTraits_t ConstraintKernelTraits(N);

    const unsigned GRID_DIM = ConstraintKernelTraits.GRID_DIM;
    const unsigned BLOCK_DIM = ConstraintKernelTraits.BLOCK_DIM;
    const unsigned NS = 0;

    __EvaluateConstraintTRJacobian__<<<GRID_DIM, BLOCK_DIM, NS, stream>>>(ecdata, N);
}

///
/// ConstraintFixPoint    =================================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintFixPoint(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                              Tensor &mt) {
    const graph::TensorBlock I = graph::SmallTensor::diagonal(1.0);

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    mt.setSubTensor(0, i * 2, I);
}

///
/// ConstraintParametrizedXfix    =========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintParametrizedXfix(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, Tensor &mt) {

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    /// wspolrzedna [X]
    mt.setValue(0, i * 2 + 0, 1.0);
}

///
/// ConstraintParametrizedYfix    =========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintParametrizedYfix(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, Tensor &mt) {

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    /// wspolrzedna [X]
    mt.setValue(0, i * 2 + 1, 1.0);
}

///
/// ConstraintConnect2Points    ===========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintConnect2Points(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                    Tensor &mt) {
    const graph::TensorBlock I = graph::SmallTensor::diagonal(1.0);
    const graph::TensorBlock mI = graph::SmallTensor::diagonal(-1.0);

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    mt.setSubTensor(0, i * 2, I);
    // l
    i = ec->pointOffset[constraint->l];
    mt.setSubTensor(0, i * 2, mI);
}

///
/// ConstraintHorizontalPoint    ==========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintHorizontalPoint(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                     Tensor &mt) {

    int i;

    i = ec->pointOffset[constraint->k];
    mt.setValue(0, i * 2, 1.0); // zero-X
    //
    i = ec->pointOffset[constraint->l];
    mt.setValue(0, i * 2, -1.0); // zero-X
}

///
/// ConstraintVerticalPoint    ============================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintVerticalPoint(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                   Tensor &mt) {

    int i;

    i = ec->pointOffset[constraint->k];
    mt.setValue(0, i * 2 + 1, 1.0); // zero-Y
    //
    i = ec->pointOffset[constraint->l];
    mt.setValue(0, i * 2 + 1, -1.0); // zero-Y
}

///
/// ConstraintLinesParallelism    =========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintLinesParallelism(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, Tensor &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    int i;

    const graph::Vector NM = n.minus(m);
    const graph::Vector LK = l.minus(k);

    // k
    i = ec->pointOffset[constraint->k];
    mt.setVector(0, i * 2, NM.pivot());
    // l
    i = ec->pointOffset[constraint->l];
    mt.setVector(0, i * 2, NM.pivot().product(-1.0));
    // m
    i = ec->pointOffset[constraint->m];
    mt.setVector(0, i * 2, LK.pivot().product(-1.0));
    // n
    i = ec->pointOffset[constraint->n];
    mt.setVector(0, i * 2, LK.pivot());
}

///
/// ConstraintLinesPerpendicular    =======================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintLinesPerpendicular(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, Tensor &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    int i;

    /// K
    i = ec->pointOffset[constraint->k];
    mt.setVector(0, i * 2, m.minus(n));
    /// L
    i = ec->pointOffset[constraint->l];
    mt.setVector(0, i * 2, m.minus(n).product(-1.0));
    /// M
    i = ec->pointOffset[constraint->m];
    mt.setVector(0, i * 2, k.minus(l));
    /// N
    i = ec->pointOffset[constraint->n];
    mt.setVector(0, i * 2, k.minus(l).product(-1.0));
}

///
/// ConstraintEqualLength    ==============================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintEqualLength(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                 Tensor &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector vLK = l.minus(k).unit();
    const graph::Vector vNM = n.minus(m).unit();

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    mt.setVector(0, i * 2, vLK.product(-1.0));
    // l
    i = ec->pointOffset[constraint->l];
    mt.setVector(0, i * 2, vLK);
    // m
    i = ec->pointOffset[constraint->m];
    mt.setVector(0, i * 2, vNM);
    // n
    i = ec->pointOffset[constraint->n];
    mt.setVector(0, i * 2, vNM.product(-1.0));
}

///
/// ConstraintParametrizedLength    =======================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintParametrizedLength(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, Tensor &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);

    const double lk = LK.length();
    const double nm = NM.length();

    const double d = ec->getParameter(constraint->paramId)->value;

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    mt.setVector(0, i * 2, LK.product(-1.0 * d / lk));
    // l
    i = ec->pointOffset[constraint->l];
    mt.setVector(0, i * 2, LK.product(1.0 * d / lk));
    // m
    i = ec->pointOffset[constraint->m];
    mt.setVector(0, i * 2, NM.product(1.0 / nm));
    // n
    i = ec->pointOffset[constraint->n];
    mt.setVector(0, i * 2, NM.product(-1.0 / nm));
}

///
/// ConstrainTangency    ==================================================================
///
template <typename Tensor>
__device__ void setJacobianConstrainTangency(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                             Tensor &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector MK = m.minus(k);
    const graph::Vector LK = l.minus(k);
    const graph::Vector ML = m.minus(l);
    const graph::Vector NM = n.minus(m);
    const double nm = NM.product(NM);
    const double lk = LK.product(LK);
    const double CRS = LK.cross(MK);

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    mt.setVector(0, i * 2, ML.pivot().product(2.0 * CRS).plus(LK.product(2.0 * nm)));
    // l
    i = ec->pointOffset[constraint->l];
    mt.setVector(0, i * 2, MK.pivot().product(-2.0 * CRS).plus(LK.product(-2.0 * nm)));
    // m
    i = ec->pointOffset[constraint->m];
    mt.setVector(0, i * 2, LK.pivot().product(2.0 * CRS).plus(NM.product(2.0 * lk)));
    // n
    i = ec->pointOffset[constraint->n];
    mt.setVector(0, i * 2, NM.product(-2.0 * lk));
}

///
/// ConstraintCircleTangency    ===========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintCircleTangency(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                    Tensor &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector vLK = l.minus(k).unit();
    const graph::Vector vNM = n.minus(m).unit();
    const graph::Vector vMK = m.minus(k).unit();

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    mt.setVector(0, i * 2, vMK.minus(vLK));
    // l
    i = ec->pointOffset[constraint->l];
    mt.setVector(0, i * 2, vLK);
    // m
    i = ec->pointOffset[constraint->m];
    mt.setVector(0, i * 2, vMK.product(-1.0).minus(vNM));
    // n
    i = ec->pointOffset[constraint->n];
    mt.setVector(0, i * 2, vNM);
}

///
/// ConstraintDistance2Points    ==========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintDistance2Points(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                     Tensor &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);

    const graph::Vector vLK = l.minus(k).unit();

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    mt.setVector(0, i * 2, vLK.product(-1.0));
    // l
    i = i = ec->pointOffset[constraint->l];
    mt.setVector(0, i * 2, vLK);
}

///
/// ConstraintDistancePointLine    ========================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintDistancePointLine(int tID, graph::Constraint const *constraint,
                                                       ComputationState *ec, Tensor &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector LK = l.minus(k);
    const graph::Vector MK = m.minus(k);

    const double d = ec->getParameter(constraint->paramId)->value;

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    mt.setVector(0, 2 * i, LK.product(d / LK.length()).minus(MK.plus(LK).pivot()));
    // l
    i = ec->pointOffset[constraint->l];
    mt.setVector(0, 2 * i, LK.product(-1.0 * d / LK.length()));
    // m
    i = ec->pointOffset[constraint->m];
    mt.setVector(0, 2 * i, LK.pivot());
}

///
/// ConstraintAngle2Lines    ==============================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintAngle2Lines(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                 Tensor &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const double d = ec->getParameter(constraint->paramId)->value;
    const double rad = toRadians(d);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);
    const graph::Vector uLKdNM = LK.unit().product(NM.length()).product(cos(rad));
    const graph::Vector uNMdLK = NM.unit().product(LK.length()).product(cos(rad));

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    mt.setVector(0, i * 2, uLKdNM.minus(NM));
    // l
    i = ec->pointOffset[constraint->l];
    mt.setVector(0, i * 2, NM.minus(uLKdNM));
    // m
    i = ec->pointOffset[constraint->m];
    mt.setVector(0, i * 2, uNMdLK.minus(LK));
    // n
    i = ec->pointOffset[constraint->n];
    mt.setVector(0, i * 2, LK.minus(uNMdLK));
}

///
/// ConstraintSetHorizontal    ============================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintSetHorizontal(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                   Tensor &mt) {
    const graph::Vector m = graph::Vector(0.0, 0.0);
    const graph::Vector n = graph::Vector(0.0, 100.0);

    int i;

    /// K
    i = ec->pointOffset[constraint->k];
    mt.setVector(0, i * 2, m.minus(n));
    /// L
    i = ec->pointOffset[constraint->l];
    mt.setVector(0, i * 2, m.minus(n).product(-1.0));
}

///
/// ConstraintSetVertical    ==============================================================
///
template <typename Tensor>
__device__ void setJacobianConstraintSetVertical(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                 Tensor &mt) {
    const graph::Vector m = graph::Vector(0.0, 0.0);
    const graph::Vector n = graph::Vector(100.0, 0.0);

    int i;

    /// K
    i = ec->pointOffset[constraint->k];
    mt.setVector(0, i * 2, m.minus(n));
    /// L
    i = ec->pointOffset[constraint->l];
    mt.setVector(0, i * 2, m.minus(n).product(-1.0));
}

///
/// ============================ CONSTRAINT HESSIAN MATRIX  ===============================
///
/**
 * (H) - ((dfi/dq)`)/dq  - upper triangular matrix A
 */

///
/// ConstraintFixPoint  ===================================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintFixPoint(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor<Layout> &mt);

///
/// ConstraintParametrizedXfix  ===========================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintParametrizedXfix(int tID, graph::Constraint const *constraint,
                                                           ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintParametrizedYfix  ===========================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintParametrizedYfix(int tID, graph::Constraint const *constraint,
                                                           ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintConnect2Points  =============================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintConnect2Points(int tID, graph::Constraint const *constraint,
                                                         ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintHorizontalPoint  ============================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintHorizontalPoint(int tID, graph::Constraint const *constraint,
                                                          ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintVerticalPoint  ==============================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintVerticalPoint(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, graph::Tensor<Layout> &mt);
///
/// ConstraintLinesParallelism  ===========================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintLinesParallelism(int tID, graph::Constraint const *constraint,
                                                           ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintLinesPerpendicular  =========================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintLinesPerpendicular(int tID, graph::Constraint const *constraint,
                                                             ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintEqualLength  ================================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintEqualLength(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintParametrizedLength  =========================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintParametrizedLength(int tID, graph::Constraint const *constraint,
                                                             ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstrainTangency  ====================================================================
///
template <typename Layout>
__device__ void setHessianTensorConstrainTangency(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                  graph::Tensor<Layout> &mt);

///
/// ConstraintCircleTangency  =============================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintCircleTangency(int tID, graph::Constraint const *constraint,
                                                         ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintDistance2Points  ============================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintDistance2Points(int tID, graph::Constraint const *constraint,
                                                          ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintDistancePointLine  ==========================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintDistancePointLine(int tID, graph::Constraint const *constraint,
                                                            ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintAngle2Lines  ================================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintAngle2Lines(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintSetHorizontal  ==============================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintSetHorizontal(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintSetVertical  ================================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintSetVertical(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor<Layout> &mt);

/// ============================ CONSTRAINT HESSIAN MATRIX  ===============================
/// <summary>
/// Evaluate Constraint Hessian Matrix  (FI)' - ((dfi/dq)`)/dq
/// </summary>
/// <param name="tId"></param>
/// <param name="ecdata"></param>
/// <param name="mt"></param>
/// <param name="N"></param>
/// <returns></returns>
template <typename Tensor>
__device__ void EvaluateConstraintHessian_Impl(int tID, ComputationState *ecdata, Tensor &mt, size_t N) {
    ComputationState *ec = static_cast<ComputationState *>(ecdata);
    /// unpack tensor for evaluation
    /// COLUMN_ORDER - tensor A
    if (tID < N) {
        const graph::Constraint *constraint = ec->getConstraint(tID);
        switch (constraint->constraintTypeId) {
        case CONSTRAINT_TYPE_ID_FIX_POINT:
            setHessianTensorConstraintFixPoint(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_XFIX:
            setHessianTensorConstraintParametrizedXfix(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_YFIX:
            setHessianTensorConstraintParametrizedYfix(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_CONNECT_2_POINTS:
            setHessianTensorConstraintConnect2Points(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_HORIZONTAL_POINT:
            setHessianTensorConstraintHorizontalPoint(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_VERTICAL_POINT:
            setHessianTensorConstraintVerticalPoint(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PARALLELISM:
            setHessianTensorConstraintLinesParallelism(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR:
            setHessianTensorConstraintLinesPerpendicular(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_EQUAL_LENGTH:
            setHessianTensorConstraintEqualLength(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_LENGTH:
            setHessianTensorConstraintParametrizedLength(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_TANGENCY:
            setHessianTensorConstrainTangency(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_CIRCLE_TANGENCY:
            setHessianTensorConstraintCircleTangency(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_2_POINTS:
            setHessianTensorConstraintDistance2Points(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_POINT_LINE:
            setHessianTensorConstraintDistancePointLine(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_ANGLE_2_LINES:
            setHessianTensorConstraintAngle2Lines(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_SET_HORIZONTAL:
            setHessianTensorConstraintSetHorizontal(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_SET_VERTICAL:
            setHessianTensorConstraintSetVertical(tID, constraint, ec, mt);
            break;
        default:
            break;
        }
        return;
    }
}

/// =======================================================================================

__global__ void __EvaluateConstraintHessian__(ComputationState *ecdata, size_t N) {
    /// Kernel Reference Addressing
    int const tId = blockDim.x * blockIdx.x + threadIdx.x;

    ComputationState *ec = static_cast<ComputationState *>(ecdata);

    ComputationMode computationMode = ec->computationMode;

#ifdef TENSOR_SPARSE_LAYOUT
    int *const INV_P = ec->INV_P;
    int const baseOffset = ec->cooWritesStiffSize + 2 * ec->cooWirtesJacobianSize; /// into coo vector
    int *const accWriteOffset = ec->accCooWriteHessianTensor;                      /// relative offset in coo
    int *const cooRowInd = ec->cooColInd;
    int *const cooColInd = ec->cooColInd;
    double *const cooVal = ec->cooVal;
    const bool intention = true;
#endif

    /// RUNTIME DISPATCH

    if (computationMode == ComputationMode::DENSE_LAYOUT) {
        graph::DenseLayout denseLayout = graph::DenseLayout(ec->dimension, 0, 0, ec->A);
        graph::Tensor<graph::DenseLayout> tensor = graph::tensorDevMem(denseLayout, 0, 0);
        ///
        EvaluateConstraintHessian_Impl(tId, ecdata, tensor, N);
        ///
        return;
    }

#ifdef TENSOR_SPARSE_LAYOUT
    if (computationMode == ComputationMode::SPARSE_LAYOUT) {
        graph::SparseLayout sparseLayout =
            graph::SparseLayout(baseOffset, accWriteOffset, cooRowInd, cooColInd, cooVal);
        graph::Tensor<graph::SparseLayout> tensorSparseLayout = graph::tensorDevMem(sparseLayout, intention);
        ///
        EvaluateConstraintHessian_Impl(tId, ecdata, tensorSparseLayout, N);
        ///
        return;
    }
    if (computationMode == ComputationMode::DIRECT_LAYOUT) {
        graph::DirectSparseLayout directLayout =
            graph::DirectSparseLayout(baseOffset, accWriteOffset, INV_P, cooRowInd, cooColInd, cooVal);
        graph::Tensor<graph::DirectSparseLayout> tensorDirect = graph::tensorDevMem(directLayout, intention);
        ///
        EvaluateConstraintHessian_Impl(tId, ecdata, tensorDirect, N);
        ///
        return;
    }
#endif

    log_error("[gpu/tensor] computation mode unknown \n");
}

/// =======================================================================================
/// <summary>
/// Evaluate Constraint Hessian Matrix  (FI)' - ((dfi/dq)`)/dq
/// </summary>
/// <param name="stream"></param>
/// <param name="ecdata"></param>
/// <param name="N"></param>
/// <returns></returns>
KERNEL_EXECUTOR void EvaluateConstraintHessian(cudaStream_t stream, ComputationState *ecdata, size_t N) {

    typedef KernelTraits<OBJECTS_PER_THREAD, DEFAULT_BLOCK_DIM> ConstraintKernelTraits_t;
    const ConstraintKernelTraits_t ConstraintKernelTraits(N);

    const unsigned GRID_DIM = ConstraintKernelTraits.GRID_DIM;
    const unsigned BLOCK_DIM = ConstraintKernelTraits.BLOCK_DIM;
    const unsigned NS = 0;

    __EvaluateConstraintHessian__<<<GRID_DIM, BLOCK_DIM, NS, stream>>>(ecdata, N);
}

///
/// ConstraintFixPoint  ===================================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintFixPoint(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor<Layout> &mt) {
    /// empty
}

///
/// ConstraintParametrizedXfix  ===========================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintParametrizedXfix(int tID, graph::Constraint const *constraint,
                                                           ComputationState *ec, graph::Tensor<Layout> &mt) {
    /// empty
}

///
/// ConstraintParametrizedYfix  ===========================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintParametrizedYfix(int tID, graph::Constraint const *constraint,
                                                           ComputationState *ec, graph::Tensor<Layout> &mt) {
    /// empty
}

///
/// ConstraintConnect2Points  =============================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintConnect2Points(int tID, graph::Constraint const *constraint,
                                                         ComputationState *ec, graph::Tensor<Layout> &mt) {
    /// empty
}

///
/// ConstraintHorizontalPoint  ============================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintHorizontalPoint(int tID, graph::Constraint const *constraint,
                                                          ComputationState *ec, graph::Tensor<Layout> &mt) {
    /// empty
}

///
/// ConstraintVerticalPoint  ==============================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintVerticalPoint(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, graph::Tensor<Layout> &mt) {
    /// empty
}

///
/// ConstraintLinesParallelism  ===========================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintLinesParallelism(int tID, graph::Constraint const *constraint,
                                                           ComputationState *ec, graph::Tensor<Layout> &mt) {
    /// macierz NxN
    const double lagrange = ec->getLagrangeMultiplier(tID);
    const graph::TensorBlock R = graph::SmallTensor::rotation(90 + 180).multiplyC(lagrange); /// R
    const graph::TensorBlock Rm = graph::SmallTensor::rotation(90).multiplyC(lagrange);      /// Rm = -R

    int i;
    int j;

    // k,m
    i = ec->pointOffset[constraint->k];
    j = ec->pointOffset[constraint->m];
    mt.plusSubTensor(2 * i, 2 * j, R);

    // k,n
    i = ec->pointOffset[constraint->k];
    j = ec->pointOffset[constraint->n];
    mt.plusSubTensor(2 * i, 2 * j, Rm);

    // l,m
    i = ec->pointOffset[constraint->l];
    j = ec->pointOffset[constraint->m];
    mt.plusSubTensor(2 * i, 2 * j, Rm);

    // l,n
    i = ec->pointOffset[constraint->l];
    j = ec->pointOffset[constraint->n];
    mt.plusSubTensor(2 * i, 2 * j, R);

    // m,k
    i = ec->pointOffset[constraint->m];
    j = ec->pointOffset[constraint->k];
    mt.plusSubTensor(2 * i, 2 * j, Rm);

    // m,l
    i = ec->pointOffset[constraint->m];
    j = ec->pointOffset[constraint->l];
    mt.plusSubTensor(2 * i, 2 * j, R);

    // n,k
    i = ec->pointOffset[constraint->n];
    j = ec->pointOffset[constraint->k];
    mt.plusSubTensor(2 * i, 2 * j, R);

    // n,l
    i = ec->pointOffset[constraint->n];
    j = ec->pointOffset[constraint->l];
    mt.plusSubTensor(2 * i, 2 * j, Rm);
}

///
/// ConstraintLinesPerpendicular  =========================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintLinesPerpendicular(int tID, graph::Constraint const *constraint,
                                                             ComputationState *ec, graph::Tensor<Layout> &mt) {
    const double lagrange = ec->getLagrangeMultiplier(tID);
    const graph::TensorBlock I = graph::SmallTensor::identity(1.0 * lagrange);
    const graph::TensorBlock Im = graph::SmallTensor::identity(1.0 * lagrange);

    int i;
    int j;

    // wstawiamy I,-I w odpowiednie miejsca
    /// K,M
    i = ec->pointOffset[constraint->k];
    j = ec->pointOffset[constraint->m];
    mt.plusSubTensor(2 * i, 2 * j, I);

    /// K,N
    i = ec->pointOffset[constraint->k];
    j = ec->pointOffset[constraint->n];
    mt.plusSubTensor(2 * i, 2 * j, Im);

    /// L,M
    i = ec->pointOffset[constraint->l];
    j = ec->pointOffset[constraint->m];
    mt.plusSubTensor(2 * i, 2 * j, Im);

    /// L,N
    i = ec->pointOffset[constraint->l];
    j = ec->pointOffset[constraint->n];
    mt.plusSubTensor(2 * i, 2 * j, I);

    /// M,K
    i = ec->pointOffset[constraint->m];
    j = ec->pointOffset[constraint->k];
    mt.plusSubTensor(2 * i, 2 * j, I);

    /// M,L
    i = ec->pointOffset[constraint->m];
    j = ec->pointOffset[constraint->l];
    mt.plusSubTensor(2 * i, 2 * j, Im);

    /// N,K
    i = ec->pointOffset[constraint->n];
    j = ec->pointOffset[constraint->k];
    mt.plusSubTensor(2 * i, 2 * j, Im);

    /// N,L
    i = ec->pointOffset[constraint->n];
    j = ec->pointOffset[constraint->l];
    mt.plusSubTensor(2 * i, 2 * j, I);
}

///
/// ConstraintEqualLength  ================================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintEqualLength(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor<Layout> &mt) {
    /// empty
}

///
/// ConstraintParametrizedLength  =========================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintParametrizedLength(int tID, graph::Constraint const *constraint,
                                                             ComputationState *ec, graph::Tensor<Layout> &mt) {
    /// empty
}

///
/// ConstrainTangency  ====================================================================
///
template <typename Layout>
__device__ void setHessianTensorConstrainTangency(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                  graph::Tensor<Layout> &mt) {
    /// equation error - java impl
}

///
/// ConstraintCircleTangency  =============================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintCircleTangency(int tID, graph::Constraint const *constraint,
                                                         ComputationState *ec, graph::Tensor<Layout> &mt) {
    /// no equation from - java impl
}

///
/// ConstraintDistance2Points  ============================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintDistance2Points(int tID, graph::Constraint const *constraint,
                                                          ComputationState *ec, graph::Tensor<Layout> &mt) {
    /// empty
}

///
/// ConstraintDistancePointLine  ==========================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintDistancePointLine(int tID, graph::Constraint const *constraint,
                                                            ComputationState *ec, graph::Tensor<Layout> &mt) {
    /// equation error - java impl
}

///
/// ConstraintAngle2Lines  ================================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintAngle2Lines(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor<Layout> &mt) {
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const double lagrange = ec->getLagrangeMultiplier(tID);

    const double d = ec->getParameter(constraint->paramId)->value;
    const double rad = toRadians(d);

    const graph::Vector LK = l.minus(k).unit();
    const graph::Vector NM = n.minus(m).unit();
    double g = LK.product(NM) * cos(rad);

    const graph::TensorBlock I_1G = graph::SmallTensor::diagonal((1 - g) * lagrange);
    const graph::TensorBlock I_Gd = graph::SmallTensor::diagonal((g - 1) * lagrange);

    int i;
    int j;

    // k,k
    i = ec->pointOffset[constraint->k];
    j = ec->pointOffset[constraint->k];
    // 0

    // k,l
    i = ec->pointOffset[constraint->k];
    j = ec->pointOffset[constraint->l];
    // 0

    // k,m
    i = ec->pointOffset[constraint->k];
    j = ec->pointOffset[constraint->m];
    mt.plusSubTensor(2 * i, 2 * j, I_1G);

    // k,n
    i = ec->pointOffset[constraint->k];
    j = ec->pointOffset[constraint->n];
    mt.plusSubTensor(2 * i, 2 * j, I_Gd);

    // l,k
    i = ec->pointOffset[constraint->l];
    j = ec->pointOffset[constraint->k];
    // 0

    // l,l
    i = ec->pointOffset[constraint->l];
    j = ec->pointOffset[constraint->l];
    // 0

    // l,m
    i = ec->pointOffset[constraint->l];
    j = ec->pointOffset[constraint->m];
    mt.plusSubTensor(2 * i, 2 * j, I_Gd);

    // l,n
    i = ec->pointOffset[constraint->l];
    j = ec->pointOffset[constraint->n];
    mt.plusSubTensor(2 * i, 2 * j, I_1G);

    // m,k
    i = ec->pointOffset[constraint->m];
    j = ec->pointOffset[constraint->k];
    mt.plusSubTensor(2 * i, 2 * j, I_1G);

    // m,l
    i = ec->pointOffset[constraint->m];
    j = ec->pointOffset[constraint->l];
    mt.plusSubTensor(2 * i, 2 * j, I_Gd);

    // m,m
    i = ec->pointOffset[constraint->m];
    j = ec->pointOffset[constraint->m];
    // 0

    // m,n
    i = ec->pointOffset[constraint->m];
    j = ec->pointOffset[constraint->n];
    // 0

    // n,k
    i = ec->pointOffset[constraint->n];
    j = ec->pointOffset[constraint->k];
    mt.plusSubTensor(2 * i, 2 * j, I_Gd);

    // n,l
    i = ec->pointOffset[constraint->n];
    j = ec->pointOffset[constraint->l];
    mt.plusSubTensor(2 * i, 2 * j, I_1G);

    // n,m
    i = ec->pointOffset[constraint->n];
    j = ec->pointOffset[constraint->m];
    // 0

    // n,n
    i = ec->pointOffset[constraint->n];
    j = ec->pointOffset[constraint->n];
    // 0
}

///
/// ConstraintSetHorizontal  ==============================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintSetHorizontal(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, graph::Tensor<Layout> &mt) {
    /// empty
}

///
/// ConstraintSetVertical  ================================================================
///
template <typename Layout>
__device__ void setHessianTensorConstraintSetVertical(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor<Layout> &mt) {
    /// empty
}

#undef DEFAULT_BLOCK_DIM