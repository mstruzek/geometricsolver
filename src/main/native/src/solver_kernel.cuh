#ifndef _SOLVER_KERNEL_CUH_
#define _SOLVER_KERNEL_CUH_

#include "cuda_runtime.h"
#include "math.h"
#include "stdio.h"

#include "model.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846 // pi
#endif

/// KERNEL#
///
/// -  HostComputation , DeviceComputation 
///

/// <summary>
/// Immutable Object
/// </summary>
struct GeometricModel
{
    int size;      /// wektor stanu
    int coffSize;  /// wspolczynniki Lagrange
    int dimension; /// N - dimension = size + coffSize

    graph::Point *points;
    graph::Geometric *geometrics;
    graph::Constraint *constraints;
    graph::Parameter *parameters;

    const int *pointOffset;       /// ---
    const int *parameterOffset;   /// paramater offs from given ID
    const int *accGeometricSize;  /// accumulative offs with geometric size evaluation function
    const int *accConstraintSize; /// accumulative offs with constraint size evaluation function
};

///  ===============================================================================

struct ComputationState
{
    int cID;     /// computatio unique id
    int info;    /// device variable cuBlas
    double norm; ///  cublasDnrm2(...)

    double *A;
    double *SV; /// State Vector  [ SV = SV + dx ] , previous task -- "lineage"
    double *dx; /// przyrosty   [ A * dx = b ]
    double *b;

    GeometricModel *model;

    __host__ __device__ graph::Vector const &getPoint(int pointId) const;

    __host__ __device__ double getLagrangeMultiplier(int constraintId) const;

    __host__ __device__ graph::Point const &getPointRef(int pointId) const;

    __host__ __device__ graph::Geometric *getGeometricObject(int geometricId) const;

    __host__ __device__ graph::Constraint *getConstraint(int constraintId) const;

    __host__ __device__ graph::Parameter *getParameter(int parameterId) const;
};

///  ===============================================================================

__host__ __device__ graph::Vector const &ComputationState::getPoint(int pointId) const
{
    int offset = model->pointOffset[pointId];
    graph::Vector *vector;
    *((void **)&vector) = &SV[offset * 2];
    return *vector;
}

__host__ __device__ double ComputationState::getLagrangeMultiplier(int constraintId) const
{
    int multiOffset = model->accConstraintSize[constraintId];
    return SV[model->size + multiOffset];
}

__host__ __device__ graph::Point const &ComputationState::getPointRef(int pointId) const
{
    int offset = model->pointOffset[pointId];
    return model->points[offset];
}

__host__ __device__ graph::Geometric *ComputationState::getGeometricObject(int geometricId) const
{
    /// geometricId is associated with `threadIdx
    return static_cast<graph::Geometric *>(&model->geometrics[geometricId]);
}

__host__ __device__ graph::Constraint *ComputationState::getConstraint(int constraintId) const
{
    /// constraintId is associated with `threadIdx
    return static_cast<graph::Constraint *>(&model->constraints[constraintId]);
}

__host__ __device__ graph::Parameter *ComputationState::getParameter(int parameterId) const
{
    int offset = model->parameterOffset[parameterId];
    return static_cast<graph::Parameter *>(&model->parameters[offset]);
}

__host__ __device__ double toRadians(double value)
{
    return (M_PI / 180.0) * value;
}

///  ===============================================================================

/// --------------- [ KERNEL# GPU ]

/// <summary>
/// Initialize State vector from actual points (without point id).
/// Reference mapping exists in Evaluation.pointOffset
/// </summary>
/// <param name="ec"></param>
/// <returns></returns>
__global__ void CopyIntoStateVector(double *SV, graph::Point *points, size_t size)
{
    int tID = blockDim.x * blockIdx.x + threadIdx.x;
    if (tID < size)
    {
        SV[2 * tID + 0] = points[tID].x;
        SV[2 * tID + 1] = points[tID].y;
    }
}

/// <summary>
/// Move computed position from State Vector into corresponding point object.
/// </summary>
/// <param name="ec"></param>
/// <returns></returns>
__global__ void CopyFromStateVector(graph::Point *points, double *SV, size_t size)
{
    int tID = blockDim.x * blockIdx.x + threadIdx.x;
    if (tID < size)
    {
        graph::Point *point = &points[tID];
        point->x = SV[2 * tID + 0];
        point->y = SV[2 * tID + 1];
    }
}

/// <summary>
/// accumulate difference from newton-raphson method;  SV[] = SV[] + b;
/// </summary>
__global__ void StateVectorAddDifference(double *SV, double *b, size_t N)
{
    int tID = blockDim.x * blockIdx.x + threadIdx.x;
    if (tID < N)
    {
        SV[2 * tID + 0] = SV[2 * tID + 0] + b[2 * tID + 0];
        SV[2 * tID + 1] = SV[2 * tID + 1] + b[2 * tID + 1];
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
__global__ void ComputeStiffnessMatrix(ComputationState *ec, int N)
{
    /// actually max single block with 1024 threads
    int tID = blockDim.x * blockIdx.x + threadIdx.x;

    // unpacka tensors from Evaluation

    const GeometricModel *model = ec->model;

    graph::Tensor tensor = graph::Tensor::fromDeviceMem(ec->A, model->dimension, model->dimension);

    if (tID < N)
    {

        const size_t rc = model->accGeometricSize[tID]; /// row-column row
        const graph::Geometric *geometric = ec->getGeometricObject(tID);

        switch (geometric->geometricTypeId)
        {

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

///
/// Free Point ============================================================================
///

__device__ void setStiffnessMatrix_FreePoint(int rc, graph::Tensor &mt)
{
    /**
     * k= I*k
     * [ -ks    ks     0;
     *    ks  -2ks   ks ;
     *     0    ks   -ks];

     */
    // K -mala sztywnosci
    graph::SmallTensor Ks = graph::SmallTensor::diagonal(CONSTS_SPRING_STIFFNESS_LOW);
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

__device__ void setStiffnessMatrix_Line(int rc, graph::Tensor &mt)
{
    /**
     * k= I*k
     * [ -ks    ks      0  	  0;
     *    ks  -ks-kb   kb  	  0;
     *     0    kb   -ks-kb   ks;
     *     0  	 0     ks    -ks];
     */
    // K -mala sztywnosci
    graph::SmallTensor Ks = graph::SmallTensor::diagonal(CONSTS_SPRING_STIFFNESS_LOW);
    // K - duza szytwnosci
    graph::SmallTensor Kb = graph::SmallTensor::diagonal(CONSTS_SPRING_STIFFNESS_HIGH);
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

__device__ void setStiffnessMatrix_FixLine(int rc, graph::Tensor &mt)
{
}

///
/// Circle ================================================================================
///

__device__ void setStiffnessMatrix_Circle(int rc, graph::Tensor &mt)
{
    /**
     * k= I*k
     * [ -ks    ks      0  	  0;
     *    ks  -ks-kb   kb  	  0;
     *     0    kb   -ks-kb   ks;
     *     0  	 0     ks    -ks];
     */
    // K -mala sztywnosci
    graph::SmallTensor Ks = graph::SmallTensor::diagonal(CONSTS_SPRING_STIFFNESS_LOW);
    // K - duza szytwnosci
    graph::SmallTensor Kb = graph::SmallTensor::diagonal(CONSTS_SPRING_STIFFNESS_HIGH * CIRCLE_SPRING_ALFA);
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

__device__ void setStiffnessMatrix_Arc(int rc, graph::Tensor &mt)
{
    // K -mala sztywnosci
    graph::SmallTensor Kb = graph::SmallTensor::diagonal(CONSTS_SPRING_STIFFNESS_HIGH);
    graph::SmallTensor Ks = graph::SmallTensor::diagonal(CONSTS_SPRING_STIFFNESS_LOW);

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
                                            graph::Tensor &mt)
{
    graph::Vector const &a = ec->getPoint(geometric->a);
    graph::Vector const &b = ec->getPoint(geometric->b);
    graph::Vector const &p1 = ec->getPoint(geometric->p1);

    double d_a_p1 = abs(p1.minus(a).length()) * 0.1;
    double d_p1_b = abs(b.minus(p1).length()) * 0.1;

    // 8 = 4*2 (4 punkty kontrolne)

    // F12 - sily w sprezynach
    graph::Vector f12 = p1.minus(a).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(p1.minus(a).length() - d_a_p1);
    // F23
    graph::Vector f23 = b.minus(p1).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(b.minus(p1).length() - d_p1_b);

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
                                       graph::Tensor &mt)
{
    graph::Vector const &a = ec->getPoint(geometric->a);
    graph::Vector const &b = ec->getPoint(geometric->b);
    graph::Vector const &p1 = ec->getPoint(geometric->p1);
    graph::Vector const &p2 = ec->getPoint(geometric->p2);

    double d_a_p1 = abs(p1.minus(a).length());
    double d_p1_p2 = abs(p2.minus(p1).length());
    double d_p2_b = abs(b.minus(p2).length());

    // 8 = 4*2 (4 punkty kontrolne)
    graph::Vector f12 = p1.minus(a)
                            .unit()
                            .product(CONSTS_SPRING_STIFFNESS_LOW)
                            .product(p1.minus(a).length() - d_a_p1); // F12 - sily w sprezynach
    graph::Vector f23 =
        p2.minus(p1).unit().product(CONSTS_SPRING_STIFFNESS_HIGH).product(p2.minus(p1).length() - d_p1_p2); // F23
    graph::Vector f34 =
        b.minus(p2).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(b.minus(p2).length() - d_p2_b); // F34

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
                                          graph::Tensor &mt)
{
}

///
/// Circle  ===============================================================================
///

__device__ void setForceIntensity_Circle(int row, graph::Geometric const *geometric, ComputationState *ec,
                                         graph::Tensor &mt)
{
    graph::Vector const &a = ec->getPoint(geometric->a);
    graph::Vector const &b = ec->getPoint(geometric->b);
    graph::Vector const &p1 = ec->getPoint(geometric->p1);
    graph::Vector const &p2 = ec->getPoint(geometric->p2);

    double d_a_p1 = abs(p1.minus(a).length());
    double d_p1_p2 = abs(p2.minus(p1).length());
    double d_p2_b = abs(b.minus(p2).length());

    // 8 = 4*2 (4 punkty kontrolne)
    graph::Vector f12 = p1.minus(a)
                            .unit()
                            .product(CONSTS_SPRING_STIFFNESS_LOW)
                            .product(p1.minus(a).length() - d_a_p1); // F12 - sily w sprezynach
    graph::Vector f23 = p2.minus(p1)
                            .unit()
                            .product(CONSTS_SPRING_STIFFNESS_HIGH * CIRCLE_SPRING_ALFA)
                            .product(p2.minus(p1).length() - d_p1_p2); // F23
    graph::Vector f34 =
        b.minus(p2).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(b.minus(p2).length() - d_p2_b); // F34

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
                                      graph::Tensor &mt)
{
    graph::Vector const &a = ec->getPoint(geometric->a);
    graph::Vector const &b = ec->getPoint(geometric->b);
    graph::Vector const &c = ec->getPoint(geometric->c);
    graph::Vector const &d = ec->getPoint(geometric->d);
    graph::Vector const &p1 = ec->getPoint(geometric->p1);
    graph::Vector const &p2 = ec->getPoint(geometric->p2);
    graph::Vector const &p3 = ec->getPoint(geometric->p3);

    /// Naciag wstepny lepiej sie zbiegaja
    double d_a_p1 = abs(p1.minus(a).length());
    double d_b_p1 = abs(p1.minus(b).length());
    double d_p1_p2 = abs(p2.minus(p1).length());
    double d_p1_p3 = abs(p3.minus(p1).length());
    double d_p3_d = abs(d.minus(p3).length());
    double d_p2_c = abs(c.minus(p2).length());

    graph::Vector fap1 = p1.minus(a).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(p1.minus(a).length() - d_a_p1);
    graph::Vector fbp1 = p1.minus(b).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(p1.minus(b).length() - d_b_p1);
    graph::Vector fp1p2 =
        p2.minus(p1).unit().product(CONSTS_SPRING_STIFFNESS_HIGH).product(p2.minus(p1).length() - d_p1_p2);
    graph::Vector fp1p3 =
        p3.minus(p1).unit().product(CONSTS_SPRING_STIFFNESS_HIGH).product(p3.minus(p1).length() - d_p1_p3);
    graph::Vector fp2c = c.minus(p2).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(c.minus(p2).length() - d_p2_c);
    graph::Vector fp3d = d.minus(p3).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(d.minus(p3).length() - d_p3_d);

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

__global__ void EvaluateForceIntensity(ComputationState *ec, int N)
{
    int tID = blockDim.x * blockIdx.x + threadIdx.x;

    const GeometricModel *model = ec->model;

    /// unpack tensor for evaluation
    graph::Tensor mt = graph::Tensor::fromDeviceMem(ec->b, model->size, 1);

    if (tID < N)
    {
        const graph::Geometric *geometric = ec->getGeometricObject(tID);

        /// row-col row
        const size_t row = model->accGeometricSize[tID];

        switch (geometric->geometricTypeId)
        {

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

///
/// ==================================== CONSTRAINT VALUE =================================
///

///
/// ConstraintFixPoint ====================================================================
///
__device__ void setValueConstraintFixPoint(int row, graph::Constraint const *constraint, ComputationState *ec,
                                           graph::Tensor &mt)
{
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector ko_vec = graph::Vector(constraint->vecX, constraint->vecY);

    ///
    graph::Vector value = k.minus(ko_vec);

    ///
    mt.setVector(row, 0, value);
}

///
/// ConstraintParametrizedXfix ============================================================
///
__device__ void setValueConstraintParametrizedXfix(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor &mt)
{
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);

    ///
    double value = k.x - param->value;
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintParametrizedYfix ============================================================
///
__device__ void setValueConstraintParametrizedYfix(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor &mt)
{
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);

    ///
    double value = k.y - param->value;
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintConnect2Points ==============================================================
///
__device__ void setValueConstraintConnect2Points(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                 graph::Tensor &mt)
{
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);

    ///
    graph::Vector value = k.minus(l);
    ///
    mt.setVector(row, 0, value);
}

///
/// ConstraintHorizontalPoint =============================================================
///
__device__ void setValueConstraintHorizontalPoint(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                  graph::Tensor &mt)
{
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);

    ///
    double value = k.x - l.x;
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintVerticalPoint ===============================================================
///
__device__ void setValueConstraintVerticalPoint(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                graph::Tensor &mt)
{
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);

    ///
    double value = k.y - l.y;
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintLinesParallelism ============================================================
///
__device__ void setValueConstraintLinesParallelism(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor &mt)
{
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);

    ///
    double value = LK.cross(NM);
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintLinesPerpendicular ==========================================================
///
__device__ void setValueConstraintLinesPerpendicular(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                     graph::Tensor &mt)
{
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector KL = k.minus(l);
    const graph::Vector MN = m.minus(n);

    ///
    double value = KL.product(MN);
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintEqualLength =================================================================
///
__device__ void setValueConstraintEqualLength(int row, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor &mt)
{
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);

    ///
    double value = LK.length() - NM.length();
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintParametrizedLength ==========================================================
///
__device__ void setValueConstraintParametrizedLength(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                     graph::Tensor &mt)
{
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);

    ///
    double value = param->value * LK.length() - NM.length();
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstrainTangency =====================================================================
///
__device__ void setValueConstrainTangency(int row, graph::Constraint const *constraint, ComputationState *ec,
                                          graph::Tensor &mt)
{
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);
    const graph::Vector MK = m.minus(k);

    ///
    double value = LK.cross(MK) * LK.cross(MK) - LK.product(LK) * NM.product(NM);
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintCircleTangency ==============================================================
///
__device__ void setValueConstraintCircleTangency(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                 graph::Tensor &mt)
{
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);
    const graph::Vector MK = m.minus(k);

    ///
    double value = LK.length() + NM.length() - MK.length();
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintDistance2Points =============================================================
///
__device__ void setValueConstraintDistance2Points(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                  graph::Tensor &mt)
{
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);

    ///
    double value = l.minus(k).length() - param->value;
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintDistancePointLine ===========================================================
///
__device__ void setValueConstraintDistancePointLine(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                    graph::Tensor &mt)
{
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);

    const graph::Vector LK = l.minus(k);
    const graph::Vector MK = m.minus(k);

    ///
    double value = LK.cross(MK) - param->value * LK.length();
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintAngle2Lines =================================================================
///
__device__ void setValueConstraintAngle2Lines(int row, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor &mt)
{
    /// coordinate system o_x axis
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);

    ///
    double value = LK.product(NM) - LK.length() * NM.length() * cos(toRadians(param->value));
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintSetHorizontal ===============================================================
///
__device__ void setValueConstraintSetHorizontal(int row, graph::Constraint const *constraint, ComputationState *ec,
                                                graph::Tensor &mt)
{
    /// coordinate system oY axis
    const graph::Vector m = graph::Vector(0.0, 0.0);
    const graph::Vector n = graph::Vector(0.0, 100.0);

    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);

    ///
    double value = k.minus(l).product(m.minus(n));
    ///
    mt.setValue(row, 0, value);
}

///
/// ConstraintSetVertical =================================================================
///
__device__ void setValueConstraintSetVertical(int row, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor &mt)
{
    /// coordinate system oX axis
    const graph::Vector m = graph::Vector(0.0, 0.0);
    const graph::Vector n = graph::Vector(100.0, 0.0);
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);

    ///
    double value = k.minus(l).product(m.minus(n));
    ///
    mt.setValue(row, 0, value);
}

///
/// Evaluate Constraint Value =============================================================
///
__global__ void EvaluateConstraintValue(ComputationState *ec, int N)
{
    int tID = blockDim.x * blockIdx.x + threadIdx.x;

    GeometricModel *model = ec->model;

    /// unpack tensor for evaluation
    graph::Tensor mt = graph::Tensor::fromDeviceMem(ec->b + (model->size), model->coffSize, 1);

    if (tID < N)
    {
        const graph::Constraint *constraint = ec->getConstraint(tID);

        const int row = model->accConstraintSize[tID];

        switch (constraint->constraintTypeId)
        {
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

///
/// ============================ CONSTRAINT JACOBIAN MATRIX  ==================================
///
/**
 * (FI)' - (dfi/dq)` transponowane - upper triangular matrix A
 */

///
/// ConstraintFixPoint    =================================================================
///
__device__ void setJacobianConstraintFixPoint(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                              graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
    const graph::Tensor I = graph::SmallTensor::diagonal(1.0);

    int i;

    // k
    i = pointOffset[constraint->k];
    mt.setSubTensor(i * 2, 0, I);
}

///
/// ConstraintParametrizedXfix    =========================================================
///
__device__ void setJacobianConstraintParametrizedXfix(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;

    int i;

    // k
    i = pointOffset[constraint->k];
    /// wspolrzedna [X]
    mt.setValue(i * 2 + 0, 0, 1.0);
}

///
/// ConstraintParametrizedYfix    =========================================================
///
__device__ void setJacobianConstraintParametrizedYfix(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;

    int i;

    // k
    i = pointOffset[constraint->k];
    /// wspolrzedna [X]
    mt.setValue(i * 2 + 1, 0, 1.0);
}

///
/// ConstraintConnect2Points    ===========================================================
///
__device__ void setJacobianConstraintConnect2Points(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                    graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
    const graph::Tensor I = graph::SmallTensor::diagonal(1.0);
    const graph::Tensor mI = graph::SmallTensor::diagonal(-1.0);

    int i;

    // k
    i = pointOffset[constraint->k];
    mt.setSubTensor(i * 2, 0, I);
    // l
    i = pointOffset[constraint->l];
    mt.setSubTensor(i * 2, 0, mI);
}

///
/// ConstraintHorizontalPoint    ==========================================================
///
__device__ void setJacobianConstraintHorizontalPoint(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                     graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;

    int i;

    i = pointOffset[constraint->k];
    mt.setValue(i * 2, 0, 1.0); // zero-X
    //
    i = pointOffset[constraint->l];
    mt.setValue(i * 2, 0, -1.0); // zero-X
}

///
/// ConstraintVerticalPoint    ============================================================
///
__device__ void setJacobianConstraintVerticalPoint(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;

    int i;

    i = pointOffset[constraint->k];
    mt.setValue(i * 2, 0, 1.0); // zero-X
    //
    i = pointOffset[constraint->l];
    mt.setValue(i * 2, 0, -1.0); // zero-X
}

///
/// ConstraintLinesParallelism    =========================================================
///
__device__ void setJacobianConstraintLinesParallelism(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    int i;

    const graph::Vector NM = n.minus(m);
    const graph::Vector LK = l.minus(k);

    // k
    i = pointOffset[constraint->k];
    mt.setVector(i * 2, 0, NM.pivot());
    // l
    i = pointOffset[constraint->l];
    mt.setVector(i * 2, 0, NM.pivot().product(-1.0));
    // m
    i = pointOffset[constraint->m];
    mt.setVector(i * 2, 0, LK.pivot().product(-1.0));
    // n
    i = pointOffset[constraint->n];
    mt.setVector(i * 2, 0, LK.pivot());
}

///
/// ConstraintLinesPerpendicular    =======================================================
///
__device__ void setJacobianConstraintLinesPerpendicular(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    int i;

    /// K
    i = pointOffset[constraint->k];
    mt.setVector(i * 2, 0, m.minus(n));
    /// L
    i = pointOffset[constraint->l];
    mt.setVector(i * 2, 0, m.minus(n).product(-1.0));
    /// M
    i = pointOffset[constraint->m];
    mt.setVector(i * 2, 0, k.minus(l));
    /// N
    i = pointOffset[constraint->n];
    mt.setVector(i * 2, 0, k.minus(l).product(-1.0));
}

///
/// ConstraintEqualLength    ==============================================================
///
__device__ void setJacobianConstraintEqualLength(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                 graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector vLK = l.minus(k).unit();
    const graph::Vector vNM = n.minus(m).unit();

    int i;

    // k
    i = pointOffset[constraint->k];
    mt.setVector(i * 2, 0, vLK.product(-1.0));
    // l
    i = pointOffset[constraint->l];
    mt.setVector(i * 2, 0, vLK);
    // m
    i = pointOffset[constraint->m];
    mt.setVector(i * 2, 0, vNM);
    // n
    i = pointOffset[constraint->n];
    mt.setVector(i * 2, 0, vNM.product(-1.0));
}

///
/// ConstraintParametrizedLength    =======================================================
///
__device__ void setJacobianConstraintParametrizedLength(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
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
    i = pointOffset[constraint->k];
    mt.setVector(i * 2, 0, LK.product(-1.0 * d / lk));
    // l
    i = pointOffset[constraint->l];
    mt.setVector(i * 2, 0, LK.product(1.0 * d / lk));
    // m
    i = pointOffset[constraint->m];
    mt.setVector(i * 2, 0, NM.product(1.0 / nm));
    // n
    i = pointOffset[constraint->n];
    mt.setVector(i * 2, 0, NM.product(-1.0 / nm));
}

///
/// ConstrainTangency    ==================================================================
///
__device__ void setJacobianConstrainTangency(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                             graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
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
    i = pointOffset[constraint->k];
    mt.setVector(i * 2, 0, ML.pivot().product(2.0 * CRS).plus(LK.product(2.0 * nm)));
    // l
    i = pointOffset[constraint->l];
    mt.setVector(i * 2, 0, MK.pivot().product(-2.0 * CRS).plus(LK.product(-2.0 * nm)));
    // m
    i = pointOffset[constraint->m];
    mt.setVector(i * 2, 0, LK.pivot().product(2.0 * CRS).plus(NM.product(2.0 * lk)));
    // n
    i = pointOffset[constraint->n];
    mt.setVector(i * 2, 0, NM.product(-2.0 * lk));
}

///
/// ConstraintCircleTangency    ===========================================================
///
__device__ void setJacobianConstraintCircleTangency(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                    graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector vLK = l.minus(k).unit();
    const graph::Vector vNM = n.minus(m).unit();
    const graph::Vector vMK = m.minus(k).unit();

    int i;

    // k
    i = pointOffset[constraint->k];
    mt.setVector(i * 2, 0, vMK.minus(vLK));
    // l
    i = pointOffset[constraint->l];
    mt.setVector(i * 2, 0, vLK);
    // m
    i = pointOffset[constraint->m];
    mt.setVector(i * 2, 0, vMK.product(-1.0).minus(vNM));
    // n
    i = pointOffset[constraint->n];
    mt.setVector(i * 2, 0, vNM);
}

///
/// ConstraintDistance2Points    ==========================================================
///
__device__ void setJacobianConstraintDistance2Points(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                     graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);

    const graph::Vector vLK = l.minus(k).unit();

    int i;

    // k
    i = pointOffset[constraint->k];
    mt.setVector(i * 2, 0, vLK.product(-1.0));
    // l
    i = i = pointOffset[constraint->l];
    mt.setVector(i * 2, 0, vLK);
}

///
/// ConstraintDistancePointLine    ========================================================
///
__device__ void setJacobianConstraintDistancePointLine(int tID, graph::Constraint const *constraint,
                                                       ComputationState *ec, graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
    const graph::Vector &k = ec->getPoint(constraint->k);
    const graph::Vector &l = ec->getPoint(constraint->l);
    const graph::Vector &m = ec->getPoint(constraint->m);
    const graph::Vector &n = ec->getPoint(constraint->n);

    const graph::Vector LK = l.minus(k);
    const graph::Vector MK = m.minus(k);

    const double d = ec->getParameter(constraint->paramId)->value;

    int i;

    // k
    i = pointOffset[constraint->k];
    mt.setVector(2 * i, 0, LK.product(d / LK.length()).minus(MK.plus(LK).pivot()));
    // l
    i = pointOffset[constraint->l];
    mt.setVector(2 * i, 0, LK.product(-1.0 * d / LK.length()));
    // m
    i = pointOffset[constraint->m];
    mt.setVector(2 * i, 0, LK.pivot());
}

///
/// ConstraintAngle2Lines    ==============================================================
///
__device__ void setJacobianConstraintAngle2Lines(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                 graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
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
    i = pointOffset[constraint->k];
    mt.setVector(i * 2, 0, uLKdNM.minus(NM));
    // l
    i = pointOffset[constraint->l];
    mt.setVector(i * 2, 0, NM.minus(uLKdNM));
    // m
    i = pointOffset[constraint->m];
    mt.setVector(i * 2, 0, uNMdLK.minus(LK));
    // n
    i = pointOffset[constraint->n];
    mt.setVector(i * 2, 0, LK.minus(uNMdLK));
}

///
/// ConstraintSetHorizontal    ============================================================
///
__device__ void setJacobianConstraintSetHorizontal(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                   graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
    const graph::Vector m = graph::Vector(0.0, 0.0);
    const graph::Vector n = graph::Vector(0.0, 100.0);

    int i;

    /// K
    i = pointOffset[constraint->k];
    mt.setVector(i * 2, 0, m.minus(n));
    /// L
    i = pointOffset[constraint->l];
    mt.setVector(i * 2, 0, m.minus(n).product(-1.0));
}

///
/// ConstraintSetVertical    ==============================================================
///
__device__ void setJacobianConstraintSetVertical(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                 graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
    const graph::Vector m = graph::Vector(0.0, 0.0);
    const graph::Vector n = graph::Vector(100.0, 0.0);

    int i;

    /// K
    i = pointOffset[constraint->k];
    mt.setVector(i * 2, 0, m.minus(n));
    /// L
    i = pointOffset[constraint->l];
    mt.setVector(i * 2, 0, m.minus(n).product(-1.0));
}

///
/// Evaluate Constraint Jacobian ==========================================================
///
///
/// (FI)' - (dfi/dq)` transponowane - upper traingular matrix A
///
///
__global__ void EvaluateConstraintJacobian(ComputationState *ec, int N)
{

    int tID = blockDim.x * blockIdx.x + threadIdx.x;

    const GeometricModel *model = ec->model;
    /// unpack tensor for evaluation
    /// COLUMN_ORDER - tensor A
    int constraintOffset = (model->size) + model->accConstraintSize[tID];

    graph::Tensor mt = graph::Tensor::fromDeviceMem(ec->A + (model->dimension) * constraintOffset, model->size, 1);

    if (tID < N)
    {

        const graph::Constraint *constraint = ec->getConstraint(tID);

        switch (constraint->constraintTypeId)
        {
        case CONSTRAINT_TYPE_ID_FIX_POINT:
            setJacobianConstraintFixPoint(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_XFIX:
            setJacobianConstraintParametrizedXfix(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_YFIX:
            setJacobianConstraintParametrizedYfix(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_CONNECT_2_POINTS:
            setJacobianConstraintConnect2Points(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_HORIZONTAL_POINT:
            setJacobianConstraintHorizontalPoint(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_VERTICAL_POINT:
            setJacobianConstraintVerticalPoint(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PARALLELISM:
            setJacobianConstraintLinesParallelism(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR:
            setJacobianConstraintLinesPerpendicular(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_EQUAL_LENGTH:
            setJacobianConstraintEqualLength(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_LENGTH:
            setJacobianConstraintParametrizedLength(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_TANGENCY:
            setJacobianConstrainTangency(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_CIRCLE_TANGENCY:
            setJacobianConstraintCircleTangency(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_2_POINTS:
            setJacobianConstraintDistance2Points(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_POINT_LINE:
            setJacobianConstraintDistancePointLine(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_ANGLE_2_LINES:
            setJacobianConstraintAngle2Lines(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_SET_HORIZONTAL:
            setJacobianConstraintSetHorizontal(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_SET_VERTICAL:
            setJacobianConstraintSetVertical(tID, constraint, ec, mt);
            break;
        default:
            break;
        }
        return;
    }
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
                                                   graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintParametrizedXfix  ===========================================================
///
__device__ void setHessianTensorConstraintParametrizedXfix(int tID, graph::Constraint const *constraint,
                                                           ComputationState *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintParametrizedYfix  ===========================================================
///
__device__ void setHessianTensorConstraintParametrizedYfix(int tID, graph::Constraint const *constraint,
                                                           ComputationState *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintConnect2Points  =============================================================
///
__device__ void setHessianTensorConstraintConnect2Points(int tID, graph::Constraint const *constraint,
                                                         ComputationState *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintHorizontalPoint  ============================================================
///
__device__ void setHessianTensorConstraintHorizontalPoint(int tID, graph::Constraint const *constraint,
                                                          ComputationState *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintVerticalPoint  ==============================================================
///
__device__ void setHessianTensorConstraintVerticalPoint(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintLinesParallelism  ===========================================================
///
__device__ void setHessianTensorConstraintLinesParallelism(int tID, graph::Constraint const *constraint,
                                                           ComputationState *ec, graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
    /// macierz NxN
    const double lagrange = ec->getLagrangeMultiplier(tID);
    const graph::Tensor R = graph::SmallTensor::rotation(90 + 180).multiplyC(lagrange); /// R
    const graph::Tensor Rm = graph::SmallTensor::rotation(90).multiplyC(lagrange);      /// Rm = -R

    int i;
    int j;

    // k,m
    i = pointOffset[constraint->k];
    j = pointOffset[constraint->m];
    mt.plusSubTensor(2 * i, 2 * j, R);

    // k,n
    i = pointOffset[constraint->k];
    j = pointOffset[constraint->n];
    mt.plusSubTensor(2 * i, 2 * j, Rm);

    // l,m
    i = pointOffset[constraint->l];
    j = pointOffset[constraint->m];
    mt.plusSubTensor(2 * i, 2 * j, Rm);

    // l,n
    i = pointOffset[constraint->l];
    j = pointOffset[constraint->n];
    mt.plusSubTensor(2 * i, 2 * j, R);

    // m,k
    i = pointOffset[constraint->m];
    j = pointOffset[constraint->k];
    mt.plusSubTensor(2 * i, 2 * j, Rm);

    // m,l
    i = pointOffset[constraint->m];
    j = pointOffset[constraint->l];
    mt.plusSubTensor(2 * i, 2 * j, R);

    // n,k
    i = pointOffset[constraint->n];
    j = pointOffset[constraint->k];
    mt.plusSubTensor(2 * i, 2 * j, R);

    // n,l
    i = pointOffset[constraint->n];
    j = pointOffset[constraint->l];
    mt.plusSubTensor(2 * i, 2 * j, Rm);
}

///
/// ConstraintLinesPerpendicular  =========================================================
///
__device__ void setHessianTensorConstraintLinesPerpendicular(int tID, graph::Constraint const *constraint,
                                                             ComputationState *ec, graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
    const double lagrange = ec->getLagrangeMultiplier(tID);
    const graph::Tensor I = graph::SmallTensor::identity(1.0 * lagrange);
    const graph::Tensor Im = graph::SmallTensor::identity(1.0 * lagrange);

    int i;
    int j;

    // wstawiamy I,-I w odpowiednie miejsca
    /// K,M
    i = pointOffset[constraint->k];
    j = pointOffset[constraint->m];
    mt.plusSubTensor(2 * i, 2 * j, I);

    /// K,N
    i = pointOffset[constraint->k];
    j = pointOffset[constraint->n];
    mt.plusSubTensor(2 * i, 2 * j, Im);

    /// L,M
    i = pointOffset[constraint->l];
    j = pointOffset[constraint->m];
    mt.plusSubTensor(2 * i, 2 * j, Im);

    /// L,N
    i = pointOffset[constraint->l];
    j = pointOffset[constraint->n];
    mt.plusSubTensor(2 * i, 2 * j, I);

    /// M,K
    i = pointOffset[constraint->m];
    j = pointOffset[constraint->k];
    mt.plusSubTensor(2 * i, 2 * j, I);

    /// M,L
    i = pointOffset[constraint->m];
    j = pointOffset[constraint->l];
    mt.plusSubTensor(2 * i, 2 * j, Im);

    /// N,K
    i = pointOffset[constraint->n];
    j = pointOffset[constraint->k];
    mt.plusSubTensor(2 * i, 2 * j, Im);

    /// N,L
    i = pointOffset[constraint->n];
    j = pointOffset[constraint->l];
    mt.plusSubTensor(2 * i, 2 * j, I);
}

///
/// ConstraintEqualLength  ================================================================
///
__device__ void setHessianTensorConstraintEqualLength(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintParametrizedLength  =========================================================
///
__device__ void setHessianTensorConstraintParametrizedLength(int tID, graph::Constraint const *constraint,
                                                             ComputationState *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstrainTangency  ====================================================================
///
__device__ void setHessianTensorConstrainTangency(int tID, graph::Constraint const *constraint, ComputationState *ec,
                                                  graph::Tensor &mt)
{
    /// equation error - java impl
}

///
/// ConstraintCircleTangency  =============================================================
///
__device__ void setHessianTensorConstraintCircleTangency(int tID, graph::Constraint const *constraint,
                                                         ComputationState *ec, graph::Tensor &mt)
{
    /// no equation from - java impl
}

///
/// ConstraintDistance2Points  ============================================================
///
__device__ void setHessianTensorConstraintDistance2Points(int tID, graph::Constraint const *constraint,
                                                          ComputationState *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintDistancePointLine  ==========================================================
///
__device__ void setHessianTensorConstraintDistancePointLine(int tID, graph::Constraint const *constraint,
                                                            ComputationState *ec, graph::Tensor &mt)
{
    /// equation error - java impl
}

///
/// ConstraintAngle2Lines  ================================================================
///
__device__ void setHessianTensorConstraintAngle2Lines(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor &mt)
{
    const int *pointOffset = ec->model->pointOffset;
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
    i = pointOffset[constraint->k];
    j = pointOffset[constraint->k];
    // 0

    // k,l
    i = pointOffset[constraint->k];
    j = pointOffset[constraint->l];
    // 0

    // k,m
    i = pointOffset[constraint->k];
    j = pointOffset[constraint->m];
    mt.plusSubTensor(2 * i, 2 * j, I_1G);

    // k,n
    i = pointOffset[constraint->k];
    j = pointOffset[constraint->n];
    mt.plusSubTensor(2 * i, 2 * j, I_Gd);

    // l,k
    i = pointOffset[constraint->l];
    j = pointOffset[constraint->k];
    // 0

    // l,l
    i = pointOffset[constraint->l];
    j = pointOffset[constraint->l];
    // 0

    // l,m
    i = pointOffset[constraint->l];
    j = pointOffset[constraint->m];
    mt.plusSubTensor(2 * i, 2 * j, I_Gd);

    // l,n
    i = pointOffset[constraint->l];
    j = pointOffset[constraint->n];
    mt.plusSubTensor(2 * i, 2 * j, I_1G);

    // m,k
    i = pointOffset[constraint->m];
    j = pointOffset[constraint->k];
    mt.plusSubTensor(2 * i, 2 * j, I_1G);

    // m,l
    i = pointOffset[constraint->m];
    j = pointOffset[constraint->l];
    mt.plusSubTensor(2 * i, 2 * j, I_Gd);

    // m,m
    i = pointOffset[constraint->m];
    j = pointOffset[constraint->m];
    // 0

    // m,n
    i = pointOffset[constraint->m];
    j = pointOffset[constraint->n];
    // 0

    // n,k
    i = pointOffset[constraint->n];
    j = pointOffset[constraint->k];
    mt.plusSubTensor(2 * i, 2 * j, I_Gd);

    // n,l
    i = pointOffset[constraint->n];
    j = pointOffset[constraint->l];
    mt.plusSubTensor(2 * i, 2 * j, I_1G);

    // n,m
    i = pointOffset[constraint->n];
    j = pointOffset[constraint->m];
    // 0

    // n,n
    i = pointOffset[constraint->n];
    j = pointOffset[constraint->n];
    // 0
}

///
/// ConstraintSetHorizontal  ==============================================================
///
__device__ void setHessianTensorConstraintSetHorizontal(int tID, graph::Constraint const *constraint,
                                                        ComputationState *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintSetVertical  ================================================================
///
__device__ void setHessianTensorConstraintSetVertical(int tID, graph::Constraint const *constraint,
                                                      ComputationState *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// Evaluate Constraint Hessian Matrix=====================================================
///
///
/// (FI)' - ((dfi/dq)`)/dq
///
///
__global__ void EvaluateConstraintHessian(ComputationState *ec, int N)
{
    int tID = blockDim.x * blockIdx.x + threadIdx.x;

    const GeometricModel *model = ec->model;
    /// unpack tensor for evaluation
    /// COLUMN_ORDER - tensor A

    graph::Tensor mt = graph::Tensor::fromDeviceMem(ec->A, model->size, 1);

    if (tID < N)
    {

        const graph::Constraint *constraint = ec->getConstraint(tID);

        switch (constraint->constraintTypeId)
        {
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
#endif // #ifndef _SOLVER_KERNEL_CUH_