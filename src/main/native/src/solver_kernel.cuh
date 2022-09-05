#ifndef _SOLVER_KERNEL_CUH_
#define _SOLVER_KERNEL_CUH_

#include "cuda_runtime.h"
#include "math.h"
#include "stdio.h"

#include "model.cuh"

#include "computation_state.cuh"

/// KERNEL#



/// ==============================================================================
///
///                             debug utility
///
/// ==============================================================================

__device__ constexpr const char *WIDEN_DOUBLE_STR_FORMAT = "%26d";
__device__ constexpr const char *FORMAT_STR_DOUBLE = " %11.2e";
__device__ constexpr const char *FORMAT_STR_IDX_DOUBLE = "%% %2d  %11.2e \n";
__device__ constexpr const char *FORMAT_STR_IDX_DOUBLE_E = "%%     %11.2e \n";
__device__ constexpr const char *FORMAT_STR_DOUBLE_CM = ", %11.2e";

__global__ void stdoutTensorData(ComputationStateData *ecdata, size_t rows, size_t cols) {

    ComputationState *ev = static_cast<ComputationState *>(ecdata);

    const graph::Layout layout = graph::defaultColumnMajor(rows, 0, 0);
    const graph::Tensor t = graph::tensorDevMem(ev->A, layout, rows, cols);

    printf("A \n");
    printf("\n MatrixDouble2 - %d x %d **************************************** \n", t.rows, t.cols);

    /// table header
    for (int i = 0; i < cols / 2; i++) {
        printf(WIDEN_DOUBLE_STR_FORMAT, i);
    }
    printf("\n");

    /// table ecdata

    for (int i = 0; i < rows; i++) {
        printf(FORMAT_STR_DOUBLE, t.getValue(i, 0));

        for (int j = 1; j < cols; j++) {
            printf(FORMAT_STR_DOUBLE_CM, t.getValue(i, j));
        }
        if (i < cols - 1)
            printf("\n");
    }
}

__global__ void stdoutRightHandSide(ComputationStateData *ecdata, size_t rows) {

    ComputationState *ev = static_cast<ComputationState *>(ecdata);

    const graph::Layout layout = graph::defaultColumnMajor(rows, 0, 0);
    const graph::Tensor b = graph::tensorDevMem(ev->b, layout, rows, 1);

    printf("\n b \n");
    printf("\n MatrixDouble1 - %d x 1 ****************************************\n", b.rows);
    printf("\n");
    /// table ecdata

    for (int i = 0; i < rows; i++) {
        printf(FORMAT_STR_DOUBLE, b.getValue(i, 0));
        printf("\n");
    }
}

__global__ void stdoutStateVector(ComputationStateData *ecdata, size_t rows) {

    ComputationState *ev = static_cast<ComputationState *>(ecdata);

    const graph::Layout layout = graph::defaultColumnMajor(rows, 0, 0);
    const graph::Tensor SV = graph::tensorDevMem(ev->SV, layout, rows, 1);

    printf("\n SV - computation ( %d ) \n", ev->cID);
    printf("\n MatrixDouble1 - %d x 1 ****************************************\n", SV.rows);
    printf("\n");
    /// table ecdata

    for (int i = 0; i < rows; i++) {
        int pointId = ev->points[i / 2].id;
        double value = SV.getValue(i, 0);
        switch (i % 2 == 0) {
        case 0:
            printf(FORMAT_STR_IDX_DOUBLE, pointId, value);
            break;
        case 1:
            printf(FORMAT_STR_IDX_DOUBLE_E, value);
            break;
        }
        printf("\n");
    }
}

///  ===============================================================================

/// --------------- [ KERNEL# GPU ]

/// <summary>
/// Initialize State vector from actual points (without point id).
/// Reference mapping exists in Evaluation.pointOffset
/// </summary>
/// <param name="ec"></param>
/// <returns></returns>
__global__ void CopyIntoStateVector(double *SV, graph::Point *points, size_t size) {
    int tID = blockDim.x * blockIdx.x + threadIdx.x;
    if (tID < size) {
        SV[2 * tID + 0] = points[tID].x;
        SV[2 * tID + 1] = points[tID].y;
    }
}

/// <summary>
/// Move computed position from State Vector into corresponding point object.
/// </summary>
/// <param name="ec"></param>
/// <returns></returns>
__global__ void CopyFromStateVector(graph::Point *points, double *SV, size_t size) {
    int tID = blockDim.x * blockIdx.x + threadIdx.x;
    if (tID < size) {
        graph::Point *point = &points[tID];
        point->x = SV[2 * tID + 0];
        point->y = SV[2 * tID + 1];
    }
}

/// <summary> CUB -- ELEMNTS_PER_THREAD ?? 
/// accumulate difference from newton-raphson method;  SV[] = SV[] + dx;
/// </summary>
__global__ void StateVectorAddDifference(double *SV, double *dx, size_t N) {
    int tID = blockDim.x * blockIdx.x + threadIdx.x;
    if (tID < N) {
        SV[2 * tID + 0] = SV[2 * tID + 0] + dx[2 * tID + 0];
        SV[2 * tID + 1] = SV[2 * tID + 1] + dx[2 * tID + 1];
    }
}

///
/// ==================================== STIFFNESS MATRIX ================================= ///
///

__device__ void setStiffnessMatrix_FreePoint(int rc, graph::Tensor &mt);
__device__ void setStiffnessMatrix_Line(int rc, graph::Tensor &mt);
__device__ void setStiffnessMatrix_Line(int rc, graph::Tensor &mt);
__device__ void setStiffnessMatrix_FixLine(int rc, graph::Tensor &mt);
__device__ void setStiffnessMatrix_Circle(int rc, graph::Tensor &mt);
__device__ void setStiffnessMatrix_Arc(int rc, graph::Tensor &mt);

/**
 * @brief Compute Stiffness Matrix on each geometric object.
 *
 * Single cuda thread is responsible for evalution of an assigned geometric object.
 *
 *
 * @param ec
 * @return __global__
 */
__device__ void ComputeStiffnessMatrix_Impl(int tID, ComputationStateData *ecdata, size_t N) {

    ComputationState *ec = static_cast<ComputationState *>(ecdata);

    /// actually max single block with 1024 threads

    const graph::Layout layout = graph::defaultColumnMajor(ec->dimension, 0, 0);

    graph::Tensor tensor = graph::tensorDevMem(ec->A, layout, ec->dimension, ec->dimension);

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
            break;
        }
    }
    return;
}

__global__ void ComputeStiffnessMatrix(ComputationStateData *ecdata, size_t N) {
    /// <summary>
    /// 
    /// </summary>
    /// <param name="ecdata"></param>
    /// <param name="N"></param>
    /// <returns></returns>
    /// 
    
    /// From Kernel Reference Addressing 
    int tID = blockDim.x * blockIdx.x + threadIdx.x;
    
    ComputeStiffnessMatrix_Impl(tID, ecdata, N);
}



/// -1 * b from equation reduction
__device__ __constant__ constexpr double SPRING_LOW = CONSTS_SPRING_STIFFNESS_LOW;

__device__ __constant__ constexpr double SPRING_HIGH = CONSTS_SPRING_STIFFNESS_HIGH;

__device__ __constant__ constexpr double SPRING_ALFA = CIRCLE_SPRING_ALFA;

///
/// Free Point ============================================================================
///

__device__ void setStiffnessMatrix_FreePoint(int rc, graph::Tensor &mt) {
    /**
     * k= I*k
     * [ -ks    ks     0;
     *    ks  -2ks   ks ;
     *     0    ks   -ks];

     */
    // K -mala sztywnosci
    graph::Tensor Ks = graph::SmallTensor::diagonal(SPRING_LOW);
    graph::Tensor Km = Ks.multiplyC(-1);

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

__device__ void setStiffnessMatrix_Line(int rc, graph::Tensor &mt) {
    /**
     * k= I*k
     * [ -ks    ks      0  	  0;
     *    ks  -ks-kb   kb  	  0;
     *     0    kb   -ks-kb   ks;
     *     0  	 0     ks    -ks];
     */
    // K -mala sztywnosci
    graph::Tensor Ks = graph::SmallTensor::diagonal(SPRING_LOW);
    // K - duza szytwnosci
    graph::Tensor Kb = graph::SmallTensor::diagonal(SPRING_HIGH);
    // -Ks-Kb
    graph::Tensor Ksb = Ks.multiplyC(-1).plus(Kb.multiplyC(-1));

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

__device__ void setStiffnessMatrix_FixLine(int rc, graph::Tensor &mt) {}

///
/// Circle ================================================================================
///

__device__ void setStiffnessMatrix_Circle(int rc, graph::Tensor &mt) {
    /**
     * k= I*k
     * [ -ks    ks      0  	  0;
     *    ks  -ks-kb   kb  	  0;
     *     0    kb   -ks-kb   ks;
     *     0  	 0     ks    -ks];
     */
    // K -mala sztywnosci
    graph::Tensor Ks = graph::SmallTensor::diagonal(SPRING_LOW);
    // K - duza szytwnosci
    graph::Tensor Kb = graph::SmallTensor::diagonal(SPRING_HIGH * CIRCLE_SPRING_ALFA);
    // -Ks-Kb
    graph::Tensor Ksb = Ks.multiplyC(-1).plus(Kb.multiplyC(-1));

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

__device__ void setStiffnessMatrix_Arc(int rc, graph::Tensor &mt) {
    // K -mala sztywnosci
    graph::Tensor Kb = graph::SmallTensor::diagonal(SPRING_HIGH);
    graph::Tensor Ks = graph::SmallTensor::diagonal(SPRING_LOW);

    graph::Tensor mKs = Ks.multiplyC(-1);
    graph::Tensor mKb = Kb.multiplyC(-1);
    graph::Tensor KsKbm = mKs.plus(mKb);

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

__device__ void setForceIntensity_FreePoint(int row, graph::Geometric const *geometric, ComputationState *ec,
                                            graph::Tensor &mt) {
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

__device__ void setForceIntensity_Line(int row, graph::Geometric const *geometric, ComputationState *ec,
                                       graph::Tensor &mt) {
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

__device__ void setForceIntensity_FixLine(int row, graph::Geometric const *geometric, ComputationState *ec,
                                          graph::Tensor &mt) {}

///
/// Circle  ===============================================================================
///

__device__ void setForceIntensity_Circle(int row, graph::Geometric const *geometric, ComputationState *ec,
                                         graph::Tensor &mt) {
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

__device__ void setForceIntensity_Arc(int row, graph::Geometric const *geometric, ComputationState *ec,
                                      graph::Tensor &mt) {
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

///
/// Evaluate Force Intensity ==============================================================
///

__device__ void EvaluateForceIntensity_Impl(int tID, ComputationStateData *ecdata, size_t N) {
    ComputationState *ec = (ComputationState *)ecdata;

    /// unpack tensor for evaluation

    const graph::Layout layout = graph::defaultColumnMajor(ec->dimension, 0, 0);

    graph::Tensor mt = graph::tensorDevMem(ec->b, layout, ec->size, 1);

    if (tID < N) {
        const graph::Geometric *geometric = ec->getGeometricObject(tID);

        const size_t row = ec->accGeometricSize[tID];

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

__global__ void EvaluateForceIntensity(ComputationStateData *ecdata, size_t N) {
    
    /// Kernel Reference Addressing
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    ///
    EvaluateForceIntensity_Impl(tId, ecdata, N);
    ///
}

///
/// ==================================== CONSTRAINT VALUE =================================
///

///
/// ConstraintFixPoint ====================================================================
///
__device__ void setValueConstraintFixPoint(int row, graph::Constraint const *constraint, ComputationState *ec,
                                           graph::Tensor &mt) {
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
__device__ void setValueConstraintParametrizedXfix(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor &mt) {
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
__device__ void setValueConstraintParametrizedYfix(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor &mt) {
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
__device__ void setValueConstraintConnect2Points(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                 graph::Tensor &mt) {
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
__device__ void setValueConstraintHorizontalPoint(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                  graph::Tensor &mt) {
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
__device__ void setValueConstraintVerticalPoint(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                graph::Tensor &mt) {
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
__device__ void setValueConstraintLinesParallelism(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor &mt) {
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
__device__ void setValueConstraintLinesPerpendicular(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                     graph::Tensor &mt) {
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
__device__ void setValueConstraintEqualLength(int row, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor &mt) {
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
__device__ void setValueConstraintParametrizedLength(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                     graph::Tensor &mt) {
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
__device__ void setValueConstrainTangency(int row, graph::Constraint const *constraint, ComputationState *ec,
                                          graph::Tensor &mt) {
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
__device__ void setValueConstraintCircleTangency(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                 graph::Tensor &mt) {
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
__device__ void setValueConstraintDistance2Points(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                  graph::Tensor &mt) {
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
__device__ void setValueConstraintDistancePointLine(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                    graph::Tensor &mt) {
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
__device__ void setValueConstraintAngle2Lines(int row, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor &mt) {
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
__device__ void setValueConstraintSetHorizontal(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                graph::Tensor &mt) {
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
__device__ void setValueConstraintSetVertical(int row, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor &mt) {
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

///
/// Evaluate Constraint Value =============================================================
///
__device__ void EvaluateConstraintValue_Impl(int tID, ComputationStateData *ecdata, size_t N) {

    ComputationState *ec = static_cast<ComputationState *>(ecdata);

    const graph::Layout layout = graph::defaultColumnMajor(ec->dimension, ec->size, 0);
    /// unpack tensor for evaluation
    graph::Tensor mt = graph::tensorDevMem(ec->b, layout, ec->coffSize, 1);

    if (tID < N) {
        const graph::Constraint *constraint = ec->getConstraint(tID);

        const int row = ec->accConstraintSize[tID];

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
            setValueConstrainTangency(row, constraint, ec, mt);
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

__global__ void EvaluateConstraintValue(ComputationStateData *ecdata, size_t N) {
    /// Kernel Reference Addressing
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    ///
    EvaluateConstraintValue_Impl(tId, ecdata, N);
    ///
}

///
/// ============================ CONSTRAINT JACOBIAN MATRIX  ==================================
///
/**
 * (FI)' - (dfi/dq)` transponowane - upper triangular matrix A
 *
 *
 *  -- templates for graph::Tensor or graph::AdapterTensor
 */

///
/// ConstraintFixPoint    =================================================================
///
template <typename Tensor = graph::Tensor>
__device__ void setJacobianConstraintFixPoint(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                              Tensor &mt) {
    const graph::Tensor I = graph::SmallTensor::diagonal(1.0);

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    mt.setSubTensor(0, i * 2, I);
}

///
/// ConstraintParametrizedXfix    =========================================================
///
template <typename Tensor = graph::Tensor>
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
template <typename Tensor = graph::Tensor>
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
template <typename Tensor = graph::Tensor>
__device__ void setJacobianConstraintConnect2Points(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                    Tensor &mt) {
    const graph::Tensor I = graph::SmallTensor::diagonal(1.0);
    const graph::Tensor mI = graph::SmallTensor::diagonal(-1.0);

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
template <typename Tensor = graph::Tensor>
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
template <typename Tensor = graph::Tensor>
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
template <typename Tensor = graph::Tensor>
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
template <typename Tensor = graph::Tensor>
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
template <typename Tensor = graph::Tensor>
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
template <typename Tensor = graph::Tensor>
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
template <typename Tensor = graph::Tensor>
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
template <typename Tensor = graph::Tensor>
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
template <typename Tensor = graph::Tensor>
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
template <typename Tensor = graph::Tensor>
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
template <typename Tensor = graph::Tensor>
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
template <typename Tensor = graph::Tensor>
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
template <typename Tensor = graph::Tensor>
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
/// Evaluate Constraint Jacobian ==========================================================
///
///
/// (FI) - (dfi/dq)   lower slice matrix of A
///
///
///
__device__ void EvaluateConstraintJacobian_Impl(int tID, ComputationStateData *ecdata, size_t N) {
    ComputationState *ec = static_cast<ComputationState *>(ecdata);

    /// unpack tensor for evaluation
    /// COLUMN_ORDER - tensor A

    if (tID < N) {
        const int constraintOffset = ec->size + ec->accConstraintSize[tID];

        const graph::Layout layout1 = graph::defaultColumnMajor(ec->dimension, constraintOffset, 0);
        graph::Tensor mt1 = graph::tensorDevMem(ec->A, layout1, 1, ec->dimension);

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

__global__ void EvaluateConstraintJacobian(ComputationStateData *ecdata, size_t N) {

    /// Kernel Reference Addressing
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    ///
    EvaluateConstraintJacobian_Impl(tId, ecdata, N);
    ///
}

///
/// Evaluate Constraint Transposed Jacobian ==========================================================
///
///
///
/// (FI)' - (dfi/dq)'   tr-transponowane - upper slice matrix  of A
///
///
__device__ void EvaluateConstraintTRJacobian_Impl(int tID, ComputationStateData *ecdata, size_t N) {

    ComputationState *ec = static_cast<ComputationState *>(ecdata);

    /// unpack tensor for evaluation
    /// COLUMN_ORDER - tensor A

    if (tID < N) {
        const int constraintOffset = ec->size + ec->accConstraintSize[tID];

        const graph::Layout layout2 =
            graph::defaultColumnMajor(ec->dimension, 0, constraintOffset); // transposed jacobian
        graph::AdapterTensor mt2 = graph::transposeTensorDevMem(ec->A, layout2, ec->dimension, 1);

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


__global__ void EvaluateConstraintTRJacobian(ComputationStateData *ecdata, size_t N) {
    
    /// Kernel Reference Addressing
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    ///
    EvaluateConstraintTRJacobian_Impl(tId, ecdata, N);
    ///

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
__device__ void setHessianTensorConstraintFixPoint(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor &mt) {
    /// empty
}

///
/// ConstraintParametrizedXfix  ===========================================================
///
__device__ void setHessianTensorConstraintParametrizedXfix(int tID, graph::Constraint const *constraint,
                                                           ComputationState *ec, graph::Tensor &mt) {
    /// empty
}

///
/// ConstraintParametrizedYfix  ===========================================================
///
__device__ void setHessianTensorConstraintParametrizedYfix(int tID, graph::Constraint const *constraint,
                                                           ComputationState *ec, graph::Tensor &mt) {
    /// empty
}

///
/// ConstraintConnect2Points  =============================================================
///
__device__ void setHessianTensorConstraintConnect2Points(int tID, graph::Constraint const *constraint,
                                                         ComputationState *ec, graph::Tensor &mt) {
    /// empty
}

///
/// ConstraintHorizontalPoint  ============================================================
///
__device__ void setHessianTensorConstraintHorizontalPoint(int tID, graph::Constraint const *constraint,
                                                          ComputationState *ec, graph::Tensor &mt) {
    /// empty
}

///
/// ConstraintVerticalPoint  ==============================================================
///
__device__ void setHessianTensorConstraintVerticalPoint(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, graph::Tensor &mt) {
    /// empty
}

///
/// ConstraintLinesParallelism  ===========================================================
///
__device__ void setHessianTensorConstraintLinesParallelism(int tID, graph::Constraint const *constraint,
                                                           ComputationState *ec, graph::Tensor &mt) {
    /// macierz NxN
    const double lagrange = ec->getLagrangeMultiplier(tID);
    const graph::Tensor R = graph::SmallTensor::rotation(90 + 180).multiplyC(lagrange); /// R
    const graph::Tensor Rm = graph::SmallTensor::rotation(90).multiplyC(lagrange);      /// Rm = -R

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
__device__ void setHessianTensorConstraintLinesPerpendicular(int tID, graph::Constraint const *constraint,
                                                             ComputationState *ec, graph::Tensor &mt) {
    const double lagrange = ec->getLagrangeMultiplier(tID);
    const graph::Tensor I = graph::SmallTensor::identity(1.0 * lagrange);
    const graph::Tensor Im = graph::SmallTensor::identity(1.0 * lagrange);

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
__device__ void setHessianTensorConstraintEqualLength(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor &mt) {
    /// empty
}

///
/// ConstraintParametrizedLength  =========================================================
///
__device__ void setHessianTensorConstraintParametrizedLength(int tID, graph::Constraint const *constraint,
                                                             ComputationState *ec, graph::Tensor &mt) {
    /// empty
}

///
/// ConstrainTangency  ====================================================================
///
__device__ void setHessianTensorConstrainTangency(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                  graph::Tensor &mt) {
    /// equation error - java impl
}

///
/// ConstraintCircleTangency  =============================================================
///
__device__ void setHessianTensorConstraintCircleTangency(int tID, graph::Constraint const *constraint,
                                                         ComputationState *ec, graph::Tensor &mt) {
    /// no equation from - java impl
}

///
/// ConstraintDistance2Points  ============================================================
///
__device__ void setHessianTensorConstraintDistance2Points(int tID, graph::Constraint const *constraint,
                                                          ComputationState *ec, graph::Tensor &mt) {
    /// empty
}

///
/// ConstraintDistancePointLine  ==========================================================
///
__device__ void setHessianTensorConstraintDistancePointLine(int tID, graph::Constraint const *constraint,
                                                            ComputationState *ec, graph::Tensor &mt) {
    /// equation error - java impl
}

///
/// ConstraintAngle2Lines  ================================================================
///
__device__ void setHessianTensorConstraintAngle2Lines(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor &mt) {
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

    const graph::Tensor I_1G = graph::SmallTensor::diagonal((1 - g) * lagrange);
    const graph::Tensor I_Gd = graph::SmallTensor::diagonal((g - 1) * lagrange);

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
__device__ void setHessianTensorConstraintSetHorizontal(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, graph::Tensor &mt) {
    /// empty
}

///
/// ConstraintSetVertical  ================================================================
///
__device__ void setHessianTensorConstraintSetVertical(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor &mt) {
    /// empty
}

///
/// Evaluate Constraint Hessian Matrix=====================================================
///
///
/// (FI)' - ((dfi/dq)`)/dq
///
///
__device__ void EvaluateConstraintHessian_Impl(int tID, ComputationStateData *ecdata, size_t N) {

    ComputationState *ec = static_cast<ComputationState *>(ecdata);

    /// unpack tensor for evaluation
    /// COLUMN_ORDER - tensor A

    const graph::Layout layout = graph::defaultColumnMajor(ec->dimension, 0, 0);

    graph::Tensor mt = graph::tensorDevMem(ec->A, layout, ec->size, ec->size);

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

__global__ void EvaluateConstraintHessian(ComputationStateData *ecdata, size_t N) {
    /// Kernel Reference Addressing
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    ///
    EvaluateConstraintHessian_Impl(tId, ecdata, N);
    ///
}



#endif // #ifndef _SOLVER_KERNEL_CUH_