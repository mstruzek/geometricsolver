#ifndef _SOLVER_KERNEL_CUH_
#define _SOLVER_KERNEL_CUH_

#include "cuda_runtime.h"
#include "math.h"
#include "stdio.h"

#include "model.cuh"

#include "tensor_layout.cuh"

#include "computation_state.cuh"

/// KERNEL#

/// ==============================================================================
///
///                             tensor utility
///
/// ==============================================================================

/// =========================================
/// Enable Tensor Sparse and Direct Layout
/// =========================================
//#define TENSOR_SPARSE_LAYOUT

#undef TENSOR_SPARSE_LAYOUT


#ifndef ELEMENTS_PER_THREAD
#define ELEMENTS_PER_THREAD 4
#endif

#define KERNEL_EXECUTOR

/// <summary>
///  Kernel Permutation compaction routine. Kernel Dispatcher.
/// </summary>
/// <param name="K_gridDim"></param>
/// <param name="K_blockDim"></param>
/// <param name="K_stream"></param>
/// <param name="nnz"></param>
/// <param name="PT1"></param>
/// <param name="PT2"></param>
/// <param name="PT"></param>
KERNEL_EXECUTOR void compactPermutationVector(unsigned K_gridDim, unsigned K_blockDim, cudaStream_t K_stream, int nnz,
                                             int *PT1, int *PT2, int *PT);



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
KERNEL_EXECUTOR void inversePermutationVector(unsigned K_gridDim, unsigned K_blockDim, cudaStream_t K_stream,
                                              int *INP, int *OUTP, size_t N); 

/// ==============================================================================
///
///                             debug utility
///
/// ==============================================================================


template <typename... Args> __device__ void log(const char *formatStr, Args... args);


template <typename... Args> __device__ void log_error(const char *formatStr, Args... args);

__global__ void stdoutTensorData(ComputationState *ecdata, size_t rows, size_t cols);

__global__ void stdoutRightHandSide(ComputationState *ecdata, size_t rows);

__global__ void stdoutStateVector(ComputationState *ecdata, size_t rows);

///  ===============================================================================

/// --------------- [ KERNEL# GPU ]

/// <summary>
/// Initialize State vector from actual points (without point id).
/// Reference mapping exists in Evaluation.pointOffset
/// </summary>
/// <param name="ec"></param>
/// <returns></returns>

__global__ void CopyIntoStateVector(double *SV, graph::Point *points, size_t size);

/// <summary>
/// Move computed position from State Vector into corresponding point object.
/// </summary>
/// <param name="ec"></param>
/// <returns></returns>
///
/// amortyzacja wzgledem inicjalizacji kernel a rejestrem watku
__global__ void CopyFromStateVector(graph::Point *points, double *SV, size_t size);

/// <summary> CUB -- ELEMNTS_PER_THREAD ??
/// accumulate difference from newton-raphson method;  SV[] = SV[] + dx;
/// </summary>

__global__ void StateVectorAddDifference(double *SV, double *dx, size_t N);

///
/// ==================================== STIFFNESS MATRIX ================================= ///
///


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
__device__ void ComputeStiffnessMatrix_Impl(int tID, ComputationState *ecdata, graph::Tensor<Layout> &tensor, size_t N);

__global__ void ComputeStiffnessMatrix(ComputationState *ecdata, size_t N);


///
/// Free Point ============================================================================
///

template<typename Layout> __device__ void setStiffnessMatrix_FreePoint(int rc, graph::Tensor<Layout> &mt);
///
/// Line ==================================================================================
///

template<typename Layout> __device__ void setStiffnessMatrix_Line(int rc, graph::Tensor<Layout> &mt);

///
/// FixLine         \\\\\\  [empty geometric]
///

template<typename Layout> __device__ void setStiffnessMatrix_FixLine(int rc, graph::Tensor<Layout> &mt);

///
/// Circle ================================================================================
///

template<typename Layout> __device__ void setStiffnessMatrix_Circle(int rc, graph::Tensor<Layout> &mt);

///
/// Arcus ================================================================================
///

template<typename Layout> __device__ void setStiffnessMatrix_Arc(int rc, graph::Tensor<Layout> &mt);

///
/// ================================ FORCE INTENSITY ==================== ///
///

///
/// Free Point ============================================================================
///

template<typename Layout>
__device__ void setForceIntensity_FreePoint(int row, graph::Geometric const *geometric, ComputationState *ec,
                                            graph::Tensor<Layout> &mt);

///
/// Line    ===============================================================================
///

template<typename Layout>
__device__ void setForceIntensity_Line(int row, graph::Geometric const *geometric, ComputationState *ec,
                                       graph::Tensor<Layout> &mt);

///
/// FixLine ===============================================================================
///

template<typename Layout>
__device__ void setForceIntensity_FixLine(int row, graph::Geometric const *geometric, ComputationState *ec,
                                          graph::Tensor<Layout> &mt);

///
/// Circle  ===============================================================================
///

template<typename Layout>
__device__ void setForceIntensity_Circle(int row, graph::Geometric const *geometric, ComputationState *ec,
                                         graph::Tensor<Layout> &mt);

///
/// Arc ===================================================================================
///

template<typename Layout>
__device__ void setForceIntensity_Arc(int row, graph::Geometric const *geometric, ComputationState *ec,
                                      graph::Tensor<Layout> &mt);

///
/// Evaluate Force Intensity ==============================================================
///

__device__ void EvaluateForceIntensity_Impl(int tID, ComputationState *ecdata, size_t N);

__global__ void EvaluateForceIntensity(ComputationState *ecdata, size_t N);

///
/// ==================================== CONSTRAINT VALUE =================================
///

///
/// ConstraintFixPoint ====================================================================
///
template<typename Layout>
__device__ void setValueConstraintFixPoint(int row, graph::Constraint const *constraint, ComputationState *ec,
                                           graph::Tensor<Layout> &mt);

///
/// ConstraintParametrizedXfix ============================================================
///
template<typename Layout>
__device__ void setValueConstraintParametrizedXfix(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor<Layout> &mt);

///
/// ConstraintParametrizedYfix ============================================================
///
template<typename Layout>
__device__ void setValueConstraintParametrizedYfix(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor<Layout> &mt);

///
/// ConstraintConnect2Points ==============================================================
///
template<typename Layout>
__device__ void setValueConstraintConnect2Points(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                 graph::Tensor<Layout> &mt);

///
/// ConstraintHorizontalPoint =============================================================
///
template<typename Layout>
__device__ void setValueConstraintHorizontalPoint(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                  graph::Tensor<Layout> &mt);

///
/// ConstraintVerticalPoint ===============================================================
///
template<typename Layout>
__device__ void setValueConstraintVerticalPoint(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                graph::Tensor<Layout> &mt);

///
/// ConstraintLinesParallelism ============================================================
///
template<typename Layout>
__device__ void setValueConstraintLinesParallelism(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor<Layout> &mt);

///
/// ConstraintLinesPerpendicular ==========================================================
///
template<typename Layout>
__device__ void setValueConstraintLinesPerpendicular(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                     graph::Tensor<Layout> &mt);

///
/// ConstraintEqualLength =================================================================
///
template<typename Layout>
__device__ void setValueConstraintEqualLength(int row, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor<Layout> &mt);

///
/// ConstraintParametrizedLength ==========================================================
///
template<typename Layout>
__device__ void setValueConstraintParametrizedLength(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                     graph::Tensor<Layout> &mt);

///
/// ConstrainTangency =====================================================================
///
template<typename Layout>
__device__ void setValueConstraintTangency(int row, graph::Constraint const *constraint, ComputationState *ec,
                                           graph::Tensor<Layout> &mt);

///
/// ConstraintCircleTangency ==============================================================
///
template<typename Layout>
__device__ void setValueConstraintCircleTangency(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                 graph::Tensor<Layout> &mt);

///
/// ConstraintDistance2Points =============================================================
///
template<typename Layout>
__device__ void setValueConstraintDistance2Points(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                  graph::Tensor<Layout> &mt);

///
/// ConstraintDistancePointLine ===========================================================
///
template<typename Layout>
__device__ void setValueConstraintDistancePointLine(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                    graph::Tensor<Layout> &mt);

///
/// ConstraintAngle2Lines =================================================================
///
template<typename Layout>
__device__ void setValueConstraintAngle2Lines(int row, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor<Layout> &mt);

///
/// ConstraintSetHorizontal ===============================================================
///
template<typename Layout>
__device__ void setValueConstraintSetHorizontal(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                graph::Tensor<Layout> &mt);

///
/// ConstraintSetVertical =================================================================
///
template<typename Layout>
__device__ void setValueConstraintSetVertical(int row, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor<Layout> &mt);
///
/// Evaluate Constraint Value =============================================================
///
__device__ void EvaluateConstraintValue_Impl(int tID, ComputationState *ecdata, size_t N);

__global__ void EvaluateConstraintValue(ComputationState *ecdata, size_t N);

///
/// ============================ CONSTRAINT JACOBIAN MATRIX  ==================================
///
/**
 * (FI)' - (dfi/dq)` transponowane - upper triangular matrix A
 *
 *
 *  -- templates for graph::TensorBlock or graph::AdapterTensor
 */

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

///
/// Evaluate Constraint Jacobian ==========================================================
///
///
/// (FI) - (dfi/dq)   lower slice matrix of A
///
///
///
template<typename Tensor>
__device__ void EvaluateConstraintJacobian_Impl(int tID, ComputationState *ecdata, Tensor &mt1, size_t N);

__global__ void EvaluateConstraintJacobian(ComputationState *ecdata, size_t N);

///
/// Evaluate Constraint Transposed Jacobian ==========================================================
///
///
///
/// (FI)' - (dfi/dq)'   tr-transponowane - upper slice matrix  of A
///
///
template<typename Tensor>
__device__ void EvaluateConstraintTRJacobian_Impl(int tID, ComputationState *ecdata, Tensor &mt2, size_t N);

__global__ void EvaluateConstraintTRJacobian(ComputationState *ecdata, size_t N);

///
/// ============================ CONSTRAINT HESSIAN MATRIX  ===============================
///
/**
 * (H) - ((dfi/dq)`)/dq  - upper triangular matrix A
 */

///
/// ConstraintFixPoint  ===================================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintFixPoint(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor<Layout> &mt);

///
/// ConstraintParametrizedXfix  ===========================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintParametrizedXfix(int tID, graph::Constraint const *constraint,
                                                           ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintParametrizedYfix  ===========================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintParametrizedYfix(int tID, graph::Constraint const *constraint,
                                                           ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintConnect2Points  =============================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintConnect2Points(int tID, graph::Constraint const *constraint,
                                                         ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintHorizontalPoint  ============================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintHorizontalPoint(int tID, graph::Constraint const *constraint,
                                                          ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintVerticalPoint  ==============================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintVerticalPoint(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, graph::Tensor<Layout> &mt);
///
/// ConstraintLinesParallelism  ===========================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintLinesParallelism(int tID, graph::Constraint const *constraint,
                                                           ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintLinesPerpendicular  =========================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintLinesPerpendicular(int tID, graph::Constraint const *constraint,
                                                             ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintEqualLength  ================================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintEqualLength(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintParametrizedLength  =========================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintParametrizedLength(int tID, graph::Constraint const *constraint,
                                                             ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstrainTangency  ====================================================================
///
template<typename Layout>
__device__ void setHessianTensorConstrainTangency(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                  graph::Tensor<Layout> &mt);

///
/// ConstraintCircleTangency  =============================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintCircleTangency(int tID, graph::Constraint const *constraint,
                                                         ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintDistance2Points  ============================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintDistance2Points(int tID, graph::Constraint const *constraint,
                                                          ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintDistancePointLine  ==========================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintDistancePointLine(int tID, graph::Constraint const *constraint,
                                                            ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintAngle2Lines  ================================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintAngle2Lines(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintSetHorizontal  ==============================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintSetHorizontal(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// ConstraintSetVertical  ================================================================
///
template<typename Layout>
__device__ void setHessianTensorConstraintSetVertical(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor<Layout> &mt);

///
/// Evaluate Constraint Hessian Matrix=====================================================
///
///
/// (FI)' - ((dfi/dq)`)/dq
///
///
template<typename Tensor>
__device__ void EvaluateConstraintHessian_Impl(int tID, ComputationState *ecdata, Tensor &mt, size_t N);

__global__ void EvaluateConstraintHessian(ComputationState *ecdata, size_t N);


#endif // #ifndef _SOLVER_KERNEL_CUH_