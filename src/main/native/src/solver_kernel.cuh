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

#define KERNEL_EXECUTOR_QQQ // przygotowac  KERNEL_EXECUTOR

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

KERNEL_EXECUTOR_QQQ __global__ void stdoutTensorData(ComputationState *ecdata, size_t rows, size_t cols);

KERNEL_EXECUTOR_QQQ __global__ void stdoutRightHandSide(ComputationState *ecdata, size_t rows);

KERNEL_EXECUTOR_QQQ __global__ void stdoutStateVector(ComputationState *ecdata, size_t rows);

///  ===============================================================================

/// --------------- [ KERNEL# GPU ]

/// <summary>
/// Initialize State vector from actual points (without point id).
/// Reference mapping exists in Evaluation.pointOffset
/// </summary>
/// <param name="ec"></param>
/// <returns></returns>



KERNEL_EXECUTOR_QQQ __global__ void CopyIntoStateVector(double *SV, graph::Point *points, size_t size);

/// <summary>
/// Move computed position from State Vector into corresponding point object.
/// </summary>
/// <param name="ec"></param>
/// <returns></returns>
///
/// amortyzacja wzgledem inicjalizacji kernel a rejestrem watku
KERNEL_EXECUTOR_QQQ __global__ void CopyFromStateVector(graph::Point *points, double *SV, size_t size);

/// <summary> CUB -- ELEMNTS_PER_THREAD ??
/// accumulate difference from newton-raphson method;  SV[] = SV[] + dx;
/// </summary>

KERNEL_EXECUTOR_QQQ __global__ void StateVectorAddDifference(double *SV, double *dx, size_t N);

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

KERNEL_EXECUTOR_QQQ __global__ void ComputeStiffnessMatrix(ComputationState *ecdata, size_t N);



///
/// ================================ FORCE INTENSITY ==================== ///
///


KERNEL_EXECUTOR_QQQ __global__ void EvaluateForceIntensity(ComputationState *ecdata, size_t N);

///
/// ==================================== CONSTRAINT VALUE =================================
///

KERNEL_EXECUTOR_QQQ __global__ void EvaluateConstraintValue(ComputationState *ecdata, size_t N);

///
/// ============================ CONSTRAINT JACOBIAN MATRIX  ==================================
///

///
/// Evaluate Constraint Jacobian ==========================================================
///
/// (FI) - (dfi/dq)   lower slice matrix of A
///


KERNEL_EXECUTOR_QQQ __global__ void EvaluateConstraintJacobian(ComputationState *ecdata, size_t N);

///
/// Evaluate Constraint Transposed Jacobian ==========================================================
///
/// (FI)' - (dfi/dq)'   tr-transponowane - upper slice matrix  of A
///
KERNEL_EXECUTOR_QQQ __global__ void EvaluateConstraintTRJacobian(ComputationState *ecdata, size_t N);


///
/// Evaluate Constraint Hessian Matrix=====================================================
///
///
/// (FI)' - ((dfi/dq)`)/dq
///
///
KERNEL_EXECUTOR_QQQ __global__ void EvaluateConstraintHessian(ComputationState *ecdata, size_t N);


#endif // #ifndef _SOLVER_KERNEL_CUH_