#ifndef _SOLVER_KERNEL_CUH_
#define _SOLVER_KERNEL_CUH_

#include "cuda_runtime.h"

#include <math.h>
#include <stdio.h>

#include "model.cuh"

/// KERNEL#

///         allocate in `setup` function  , all references are valid GPU memory references
///
///         cudaMalloc, cudaMemcpy
///
struct Computation {

    int size;      /// wektor stanu
    int coffSize;  /// wspolczynniki Lagrange
    int dimension; /// N - dimension = size + coffSize

    /// macierze ukladu rowa�
    double *A;
    double *SV; /// State Vector  [ SV = SV + dx ]
    double *dx; /// przyrosty   [ A * dx = b ]
    double *b;

    /// mainly static data
    graph::Point *points;
    graph::Geometric *geometrics;
    graph::Constraint *constraints;
    graph::Parameter *parameters;

    const int *pointOffset;       /// ---
    const int *parameterOffset;   /// paramater offset from given ID
    const int *accGeometricSize;  /// accumulative offset with geometric size evaluation function
    const int *accConstraintSize; /// accumulative offset with constraint size evaluation function

    /// w drugiej fazie dopisac => dodatkowo potrzebny per block  __shared__  memory   ORR    L1   fast region memory

    __host__ __device__ graph::Point const &getPoint(int pointId) const;

    __host__ __device__ graph::Vector const &Computation::getPointSV(int pointId) const;

    __host__ __device__ graph::Geometric *getGeomtricObject(int geometricId) const;

    __host__ __device__ graph::Constraint *getConstraint(int constraintId) const;

    __host__ __device__ graph::Parameter *getParameter(int parameterId) const;
};

__host__ __device__ graph::Point const &Computation::getPoint(int pointId) const {
    int offset = pointOffset[pointId];
    return points[offset];
};

__host__ __device__ graph::Vector const &Computation::getPointSV(int pointId) const {
    int offset = pointOffset[pointId];
    graph::Vector *vector;
    *((void **)&vector) = &SV[offset * 2];
    return *vector;
}

__host__ __device__ graph::Geometric *Computation::getGeomtricObject(int geometricId) const {
    /// geometricId is associated with `threadIdx
    return static_cast<graph::Geometric *>(&geometrics[geometricId]);
}

__host__ __device__ graph::Constraint *Computation::getConstraint(int constraintId) const {
    /// constraintId is associated with `threadIdx
    return static_cast<graph::Constraint *>(&constraints[constraintId]);
}

__host__ __device__ graph::Parameter *Computation::getParameter(int parameterId) const {
    int offset = parameterOffset[parameterId];
    return static_cast<graph::Parameter *>(&parameters[offset]);
}

__host__ __device__ double toRadians(double value) { return (M_PI / 180.0) * value; }

/// [ KERNEL# GPU ]

/// <summary>
/// Initialize State vector from actual points (without point id).
/// Reference mapping exists in Evaluation.pointOffset
/// </summary>
/// <param name="ec"></param>
/// <returns></returns>
__global__ void CopyIntoStateVector(Computation *ec) {
    int tID = blockDim.x * blockIdx.x + threadIdx.x;
    int size = ec->size;
    if (tID < size) {
        ec->SV[2 * tID + 0] = ec->points[tID].x;
        ec->SV[2 * tID + 1] = ec->points[tID].y;
    }
}

/// <summary>
/// Move computed position from State Vector into corresponding point object.
/// </summary>
/// <param name="ec"></param>
/// <returns></returns>
__global__ void CopyFromStateVector(Computation *ec) {
    int tID = blockDim.x * blockIdx.x + threadIdx.x;
    int size = ec->size;
    if (tID < size) {
        graph::Point *point = &ec->points[tID];
        point->x = ec->SV[2 * tID + 0];
        point->y = ec->SV[2 * tID + 1];
    }
}

/// <summary>
/// accumulate difference from newton-raphson method;  SV[] = SV[] + b;
/// </summary>
__global__ void StateVectorAddDifference(double *SV, double *b, size_t N) {
    int tID = blockDim.x * blockIdx.x + threadIdx.x;
    if (tID < N) {
        SV[2 * tID + 0] = SV[2 * tID + 0] + b[2 * tID + 0];
        SV[2 * tID + 1] = SV[2 * tID + 1] + b[2 * tID + 1];
    }
}

__device__ void SetupPointAccessor(Computation *ec, int GN, /* __shared__*/ double *_points) {

    int offset;

    int tID = blockDim.x * blockIdx.x + threadIdx.x;

    /// set-aside - persisten access in L2 cache !
    /// more than 1024 points will not share this reference

    graph::Tensor tensor; // view into Evaluation Context reference

    if (tID < GN) {
        graph::Geometric *geometric = ec->getGeomtricObject(tID);

        offset = ec->accGeometricSize[tID];

        switch (geometric->geometricTypeId) {

        case GEOMETRIC_TYPE_ID_FREE_POINT: {
            int aID = geometric->a;
            int bID = geometric->b;
            int p1ID = geometric->p1;

            graph::Point const &a = ec->getPoint(aID);
            graph::Point const &b = ec->getPoint(bID);
            graph::Point const &p1 = ec->getPoint(p1ID);
            /// ======================================================///
            _points[offset + 0] = a.x;
            _points[offset + 1] = a.y;

            _points[offset + 2] = p1.x;
            _points[offset + 3] = p1.y;

            _points[offset + 4] = b.x;
            _points[offset + 5] = b.y;
            /// ======================================================///
        } break;

        case GEOMETRIC_TYPE_ID_LINE: {

            int aID = geometric->a;
            int bID = geometric->b;
            int p1ID = geometric->p1;
            int p2ID = geometric->p2;

            graph::Point const &a = ec->getPoint(aID);
            graph::Point const &b = ec->getPoint(bID);
            graph::Point const &p1 = ec->getPoint(p1ID);
            graph::Point const &p2 = ec->getPoint(p2ID);
            /// ======================================================///
            _points[offset + 0] = a.x;
            _points[offset + 1] = a.y;

            _points[offset + 2] = p1.x;
            _points[offset + 3] = p1.y;

            _points[offset + 4] = p2.x;
            _points[offset + 5] = p2.y;

            _points[offset + 6] = b.x;
            _points[offset + 7] = b.y;
            /// ======================================================///
        } break;

        case GEOMETRIC_TYPE_ID_FIX_LINE:

            /// ======================================================///
            break;

        case GEOMETRIC_TYPE_ID_CIRCLE: {

            int aID = geometric->a;
            int bID = geometric->b;
            int p1ID = geometric->p1;
            int p2ID = geometric->p2;

            graph::Point const &a = ec->getPoint(aID);
            graph::Point const &b = ec->getPoint(bID);
            graph::Point const &p1 = ec->getPoint(p1ID);
            graph::Point const &p2 = ec->getPoint(p2ID);
            /// ======================================================///
            _points[offset + 0] = a.x;
            _points[offset + 1] = a.y;

            _points[offset + 2] = p1.x;
            _points[offset + 3] = p1.y;

            _points[offset + 4] = p2.x;
            _points[offset + 5] = p2.y;

            _points[offset + 6] = b.x;
            _points[offset + 7] = b.y;
            /// ======================================================///
        } break;

        case GEOMETRIC_TYPE_ID_ARC: {

            int aID = geometric->a;
            int bID = geometric->b;
            int cID = geometric->c;
            int dID = geometric->d;
            int p1ID = geometric->p1;
            int p2ID = geometric->p2;
            int p3ID = geometric->p3;

            graph::Point const &a = ec->getPoint(aID);
            graph::Point const &b = ec->getPoint(bID);
            graph::Point const &c = ec->getPoint(cID);
            graph::Point const &d = ec->getPoint(dID);
            graph::Point const &p1 = ec->getPoint(p1ID);
            graph::Point const &p2 = ec->getPoint(p2ID);
            graph::Point const &p3 = ec->getPoint(p3ID);
            /// ======================================================///
            _points[offset + 0] = a.x;
            _points[offset + 1] = a.y;

            _points[offset + 2] = b.x;
            _points[offset + 3] = b.y;

            _points[offset + 4] = c.x;
            _points[offset + 5] = c.y;

            _points[offset + 6] = d.x;
            _points[offset + 7] = d.y;

            _points[offset + 8] = p1.x;
            _points[offset + 9] = p1.y;

            _points[offset + 10] = p2.x;
            _points[offset + 11] = p2.y;

            _points[offset + 12] = p3.x;
            _points[offset + 13] = p3.y;
            /// ======================================================///

        } break;

        default:
            break;
        }
    }
}

///
/// ==================================== STIFFNESS MATRIX ================================= ///
///

__device__ void setStiffnessMatrix_FreePoint(int tID, graph::Tensor &mt);
__device__ void setStiffnessMatrix_Line(int tID, graph::Tensor &mt);
__device__ void setStiffnessMatrix_Line(int tID, graph::Tensor &mt);
__device__ void setStiffnessMatrix_FixLine(int tid, graph::Tensor &mt);
__device__ void setStiffnessMatrix_Circle(int tID, graph::Tensor &mt);
__device__ void setStiffnessMatrix_Arc(int tID, graph::Tensor &mt);

/**
 * @brief Compute Stiffness Matrix on each geometric object.
 *
 * Single cuda thread is responsible for evalution of an assigned geometric object.
 *
 *
 * @param ec
 * @return __global__
 */
__global__ void computeStiffnessMatrix(Computation *ec, int N) {
    /// actually max single block with 1024 threads
    int tID = blockDim.x * blockIdx.x + threadIdx.x;

    // unpacka tensors from Evaluation

    graph::Tensor tensor = graph::Tensor::fromDeviceMem(ec->A, ec->dimension, ec->dimension);

    if (tID < N) {
        graph::Geometric *geometric = ec->getGeomtricObject(tID);

        switch (geometric->geometricTypeId) {

        case GEOMETRIC_TYPE_ID_FREE_POINT:
            setStiffnessMatrix_FreePoint(tID, tensor);
            break;

        case GEOMETRIC_TYPE_ID_LINE:
            setStiffnessMatrix_Line(tID, tensor);
            break;

        case GEOMETRIC_TYPE_ID_FIX_LINE:
            setStiffnessMatrix_FixLine(tID, tensor);
            break;

        case GEOMETRIC_TYPE_ID_CIRCLE:
            setStiffnessMatrix_Circle(tID, tensor);
            break;

        case GEOMETRIC_TYPE_ID_ARC:
            setStiffnessMatrix_Arc(tID, tensor);
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

__device__ void setStiffnessMatrix_FreePoint(int tID, graph::Tensor &mt) {
    /**
     * k= I*k
     * [ -ks    ks     0;
     *    ks  -2ks   ks ;
     *     0    ks   -ks];

     */
    // K -mala sztywnosci
    graph::SmallTensor Ks = graph::SmallTensor::diagonal(CONSTS_SPRING_STIFFNESS_LOW);
    graph::Tensor Km = Ks.multiplyC(-1);

    mt.plusSubTensor(tID + 0, tID + 0, Km);
    mt.plusSubTensor(tID + 0, tID + 2, Ks);

    mt.plusSubTensor(tID + 2, tID + 0, Ks);
    mt.plusSubTensor(tID + 2, tID + 2, Km.multiplyC(2.0));
    mt.plusSubTensor(tID + 2, tID + 4, Ks);

    mt.plusSubTensor(tID + 4, tID + 2, Ks);
    mt.plusSubTensor(tID + 4, tID + 4, Km);
}

///
/// Line ==================================================================================
///

__device__ void setStiffnessMatrix_Line(int tID, graph::Tensor &mt) {
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
    mt.plusSubTensor(tID + 0, tID + 0, Ks.multiplyC(-1));
    mt.plusSubTensor(tID + 0, tID + 2, Ks);
    mt.plusSubTensor(tID + 2, tID + 0, Ks);
    mt.plusSubTensor(tID + 2, tID + 2, Ksb);
    mt.plusSubTensor(tID + 2, tID + 4, Kb);
    mt.plusSubTensor(tID + 4, tID + 2, Kb);
    mt.plusSubTensor(tID + 4, tID + 4, Ksb);
    mt.plusSubTensor(tID + 4, tID + 6, Ks);
    mt.plusSubTensor(tID + 6, tID + 4, Ks);
    mt.plusSubTensor(tID + 6, tID + 6, Ks.multiplyC(-1));
}

///
/// FixLine         \\\\\\  [empty geometric]
///

__device__ void setStiffnessMatrix_FixLine(int tid, graph::Tensor &mt) {}

///
/// Circle ================================================================================
///

__device__ void setStiffnessMatrix_Circle(int tID, graph::Tensor &mt) {
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
    mt.plusSubTensor(tID + 0, tID + 0, Ks.multiplyC(-1));
    mt.plusSubTensor(tID + 0, tID + 2, Ks);
    mt.plusSubTensor(tID + 2, tID + 0, Ks);
    mt.plusSubTensor(tID + 2, tID + 2, Ksb);
    mt.plusSubTensor(tID + 2, tID + 4, Kb);
    mt.plusSubTensor(tID + 4, tID + 2, Kb);
    mt.plusSubTensor(tID + 4, tID + 4, Ksb);
    mt.plusSubTensor(tID + 4, tID + 6, Ks);
    mt.plusSubTensor(tID + 6, tID + 4, Ks);
    mt.plusSubTensor(tID + 6, tID + 6, Ks.multiplyC(-1));
}

///
/// Arcus ================================================================================
///

__device__ void setStiffnessMatrix_Arc(int tID, graph::Tensor &mt) {
    // K -mala sztywnosci
    graph::SmallTensor Kb = graph::SmallTensor::diagonal(CONSTS_SPRING_STIFFNESS_HIGH);
    graph::SmallTensor Ks = graph::SmallTensor::diagonal(CONSTS_SPRING_STIFFNESS_LOW);

    graph::Tensor mKs = Ks.multiplyC(-1);
    graph::Tensor mKb = Kb.multiplyC(-1);
    graph::Tensor KsKbm = mKs.plus(mKb);

    mt.plusSubTensor(tID + 0, tID + 0, mKs);
    mt.plusSubTensor(tID + 0, tID + 8, Ks); // a

    mt.plusSubTensor(tID + 2, tID + 2, mKs);
    mt.plusSubTensor(tID + 2, tID + 8, Ks); // b

    mt.plusSubTensor(tID + 4, tID + 4, mKs);
    mt.plusSubTensor(tID + 4, tID + 10, Ks); // c

    mt.plusSubTensor(tID + 6, tID + 6, mKs);
    mt.plusSubTensor(tID + 6, tID + 12, Ks); // d

    mt.plusSubTensor(tID + 8, tID + 0, Ks);
    mt.plusSubTensor(tID + 8, tID + 2, Ks);
    mt.plusSubTensor(tID + 8, tID + 8, KsKbm.multiplyC(2.0));
    mt.plusSubTensor(tID + 8, tID + 10, Kb);
    mt.plusSubTensor(tID + 8, tID + 12, Kb); // p1

    mt.plusSubTensor(tID + 10, tID + 4, Ks);
    mt.plusSubTensor(tID + 10, tID + 8, Kb);
    mt.plusSubTensor(tID + 10, tID + 10, KsKbm); // p2

    mt.plusSubTensor(tID + 12, tID + 6, Ks);
    mt.plusSubTensor(tID + 12, tID + 8, Kb);
    mt.plusSubTensor(tID + 12, tID + 12, KsKbm); // p3
}

///
/// ================================ FORCE INTENSITY ==================== ///
///

///
/// Free Point ============================================================================
///

__device__ void evaluateForceIntensity_FreePoint(int tID, graph::Geometric *geometric, Computation *ec, graph::Tensor &mt) {
    graph::Point const &a = ec->getPoint(geometric->a);
    graph::Point const &b = ec->getPoint(geometric->b);
    graph::Point const &p1 = ec->getPoint(geometric->p1);

    double d_a_p1 = abs(p1.minus(a).length()) * 0.1;
    double d_p1_b = abs(b.minus(p1).length()) * 0.1;

    // 8 = 4*2 (4 punkty kontrolne)

    // F12 - sily w sprezynach
    graph::Vector f12 = p1.minus(a).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(p1.minus(a).length() - d_a_p1);
    // F23
    graph::Vector f23 = b.minus(p1).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(b.minus(p1).length() - d_p1_b);

    // F1 - sily na poszczegolne punkty
    mt.setVector(tID + 0, 0, f12);
    // F2
    mt.setVector(tID + 2, 0, f23.minus(f12));
    // F3
    mt.setVector(tID + 4, 0, f23.product(-1));
}

///
/// Line    ===============================================================================
///

__device__ void evaluateForceIntensity_Line(int tID, graph::Geometric *geometric, Computation *ec, graph::Tensor &mt) {
    graph::Point const &a = ec->getPoint(geometric->a);
    graph::Point const &b = ec->getPoint(geometric->b);
    graph::Point const &p1 = ec->getPoint(geometric->p1);
    graph::Point const &p2 = ec->getPoint(geometric->p2);

    double d_a_p1 = abs(p1.minus(a).length());
    double d_p1_p2 = abs(p2.minus(p1).length());
    double d_p2_b = abs(b.minus(p2).length());

    // 8 = 4*2 (4 punkty kontrolne)
    graph::Vector f12 =
        p1.minus(a).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(p1.minus(a).length() - d_a_p1); // F12 - sily w sprezynach
    graph::Vector f23 = p2.minus(p1).unit().product(CONSTS_SPRING_STIFFNESS_HIGH).product(p2.minus(p1).length() - d_p1_p2); // F23
    graph::Vector f34 = b.minus(p2).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(b.minus(p2).length() - d_p2_b);     // F34

    // F1 - silu na poszczegolne punkty
    mt.setVector(tID + 0, 0, f12);
    // F2
    mt.setVector(tID + 2, 0, f23.minus(f12));
    // F3
    mt.setVector(tID + 4, 0, f34.minus(f23));
    // F4
    mt.setVector(tID + 6, 0, f34.product(-1.0));
}

///
/// FixLine ===============================================================================
///

__device__ void evaluateForceIntensity_FixLine(int tID, graph::Geometric *geometric, Computation *ec, graph::Tensor &mt) {}

///
/// Circle  ===============================================================================
///

__device__ void evaluateForceIntensity_Circle(int tID, graph::Geometric *geometric, Computation *ec, graph::Tensor &mt) {
    graph::Point const &a = ec->getPoint(geometric->a);
    graph::Point const &b = ec->getPoint(geometric->b);
    graph::Point const &p1 = ec->getPoint(geometric->p1);
    graph::Point const &p2 = ec->getPoint(geometric->p2);

    double d_a_p1 = abs(p1.minus(a).length());
    double d_p1_p2 = abs(p2.minus(p1).length());
    double d_p2_b = abs(b.minus(p2).length());

    // 8 = 4*2 (4 punkty kontrolne)
    graph::Vector f12 =
        p1.minus(a).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(p1.minus(a).length() - d_a_p1); // F12 - sily w sprezynach
    graph::Vector f23 =
        p2.minus(p1).unit().product(CONSTS_SPRING_STIFFNESS_HIGH * CIRCLE_SPRING_ALFA).product(p2.minus(p1).length() - d_p1_p2); // F23
    graph::Vector f34 = b.minus(p2).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(b.minus(p2).length() - d_p2_b);          // F34

    // F1 - silu na poszczegolne punkty
    mt.setVector(tID + 0, 0, f12);
    // F2
    mt.setVector(tID + 2, 0, f23.minus(f12));
    // F3
    mt.setVector(tID + 4, 0, f34.minus(f23));
    // F4
    mt.setVector(tID + 6, 0, f34.product(-1.0));
}

///
/// Arc ===================================================================================
///

__device__ void evaluateForceIntensity_Arc(int tID, graph::Geometric *geometric, Computation *ec, graph::Tensor &mt) {
    graph::Point const &a = ec->getPoint(geometric->a);
    graph::Point const &b = ec->getPoint(geometric->b);
    graph::Point const &c = ec->getPoint(geometric->c);
    graph::Point const &d = ec->getPoint(geometric->d);
    graph::Point const &p1 = ec->getPoint(geometric->p1);
    graph::Point const &p2 = ec->getPoint(geometric->p2);
    graph::Point const &p3 = ec->getPoint(geometric->p3);

    /// Naciag wstepny lepiej sie zbiegaja
    double d_a_p1 = abs(p1.minus(a).length());
    double d_b_p1 = abs(p1.minus(b).length());
    double d_p1_p2 = abs(p2.minus(p1).length());
    double d_p1_p3 = abs(p3.minus(p1).length());
    double d_p3_d = abs(d.minus(p3).length());
    double d_p2_c = abs(c.minus(p2).length());

    graph::Vector fap1 = p1.minus(a).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(p1.minus(a).length() - d_a_p1);
    graph::Vector fbp1 = p1.minus(b).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(p1.minus(b).length() - d_b_p1);
    graph::Vector fp1p2 = p2.minus(p1).unit().product(CONSTS_SPRING_STIFFNESS_HIGH).product(p2.minus(p1).length() - d_p1_p2);
    graph::Vector fp1p3 = p3.minus(p1).unit().product(CONSTS_SPRING_STIFFNESS_HIGH).product(p3.minus(p1).length() - d_p1_p3);
    graph::Vector fp2c = c.minus(p2).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(c.minus(p2).length() - d_p2_c);
    graph::Vector fp3d = d.minus(p3).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(d.minus(p3).length() - d_p3_d);

    mt.setVector(tID + 0, 0, fap1);
    mt.setVector(tID + 2, 0, fbp1);
    mt.setVector(tID + 4, 0, fp2c.product(-1));
    mt.setVector(tID + 6, 0, fp3d.product(-1));
    mt.setVector(tID + 8, 0, fp1p2.plus(fp1p3).minus(fap1).minus(fbp1));
    mt.setVector(tID + 10, 0, fp2c.minus(fp1p2));
    mt.setVector(tID + 12, 0, fp3d.minus(fp1p3));
}

///
/// Evaluate Force Intensity ==============================================================
///

__global__ void evaluateForceIntensity(Computation *ec, int N) {
    int tID;
    graph::Geometric *geometric = nullptr;

    tID = blockDim.x * blockIdx.x + threadIdx.x;

    /// unpack tensor for evaluation
    graph::Tensor mt = graph::Tensor::fromDeviceMem(ec->b, ec->size, 1);

    if (tID < N) {
        geometric = ec->getGeomtricObject(tID);

        switch (geometric->geometricTypeId) {

        case GEOMETRIC_TYPE_ID_FREE_POINT:
            evaluateForceIntensity_FreePoint(tID, geometric, ec, mt);
            break;

        case GEOMETRIC_TYPE_ID_LINE:
            evaluateForceIntensity_Line(tID, geometric, ec, mt);
            break;

        case GEOMETRIC_TYPE_ID_FIX_LINE:
            evaluateForceIntensity_FixLine(tID, geometric, ec, mt);
            break;

        case GEOMETRIC_TYPE_ID_CIRCLE:
            evaluateForceIntensity_Circle(tID, geometric, ec, mt);
            break;

        case GEOMETRIC_TYPE_ID_ARC:
            evaluateForceIntensity_Arc(tID, geometric, ec, mt);
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
__device__ void evaluateValueConstraintFixPoint(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Vector ko_vec = graph::Vector(constraint->vecX, constraint->vecY);
    const int offset = ec->accConstraintSize[tID];

    ///
    graph::Vector value = k.minus(ko_vec);

    ///
    mt.setVector(offset, 0, value);
}

///
/// ConstraintParametrizedXfix ============================================================
///
__device__ void evaluateValueConstraintParametrizedXfix(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);
    const int offset = ec->accConstraintSize[tID];

    ///
    double value = k.x - param->value;
    ///
    mt.setValue(offset, 0, value);
}

///
/// ConstraintParametrizedYfix ============================================================
///
__device__ void evaluateValueConstraintParametrizedYfix(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);
    const int offset = ec->accConstraintSize[tID];

    ///
    double value = k.y - param->value;
    ///
    mt.setValue(offset, 0, value);
}

///
/// ConstraintConnect2Points ==============================================================
///
__device__ void evaluateValueConstraintConnect2Points(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const int offset = ec->accConstraintSize[tID];

    ///
    graph::Vector value = k.minus(l);
    ///
    mt.setVector(offset, 0, value);
}

///
/// ConstraintHorizontalPoint =============================================================
///
__device__ void evaluateValueConstraintHorizontalPoint(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const int offset = ec->accConstraintSize[tID];

    ///
    double value = k.x - l.x;
    ///
    mt.setValue(offset, 0, value);
}

///
/// ConstraintVerticalPoint ===============================================================
///
__device__ void evaluateValueConstraintVerticalPoint(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const int offset = ec->accConstraintSize[tID];

    ///
    double value = k.y - l.y;
    ///
    mt.setValue(offset, 0, value);
}

///
/// ConstraintLinesParallelism ============================================================
///
__device__ void evaluateValueConstraintLinesParallelism(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);
    const int offset = ec->accConstraintSize[tID];

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);

    ///
    double value = LK.cross(NM);
    ///
    mt.setValue(offset, 0, value);
}

///
/// ConstraintLinesPerpendicular ==========================================================
///
__device__ void evaluateValueConstraintLinesPerpendicular(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);
    const int offset = ec->accConstraintSize[tID];

    const graph::Vector KL = k.minus(l);
    const graph::Vector MN = m.minus(n);

    ///
    double value = KL.product(MN);
    ///
    mt.setValue(offset, 0, value);
}

///
/// ConstraintEqualLength =================================================================
///
__device__ void evaluateValueConstraintEqualLength(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);
    const int offset = ec->accConstraintSize[tID];

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);

    ///
    double value = LK.length() - NM.length();
    ///
    mt.setValue(offset, 0, value);
}

///
/// ConstraintParametrizedLength ==========================================================
///
__device__ void evaluateValueConstraintParametrizedLength(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);
    const int offset = ec->accConstraintSize[tID];

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);

    ///
    double value = param->value * LK.length() - NM.length();
    ///
    mt.setValue(offset, 0, value);
}

///
/// ConstrainTangency =====================================================================
///
__device__ void evaluateValueConstrainTangency(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);    
    const int offset = ec->accConstraintSize[tID];

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);
    const graph::Vector MK = m.minus(k);

    ///
    double value = LK.cross(MK) * LK.cross(MK) - LK.product(LK) * NM.product(NM);
    ///
    mt.setValue(offset, 0, value);
}

///
/// ConstraintCircleTangency ==============================================================
///
__device__ void evaluateValueConstraintCircleTangency(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);
    const int offset = ec->accConstraintSize[tID];

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);
    const graph::Vector MK = m.minus(k);

    ///
    double value = LK.length() + NM.length() - MK.length();
    ///
    mt.setValue(offset, 0, value);
}

///
/// ConstraintDistance2Points =============================================================
///
__device__ void evaluateValueConstraintDistance2Points(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);
    const int offset = ec->accConstraintSize[tID];

    ///
    double value = l.minus(k).length() - param->value;
    ///
    mt.setValue(offset, 0, value);
}

///
/// ConstraintDistancePointLine ===========================================================
///
__device__ void evaluateValueConstraintDistancePointLine(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);
    const int offset = ec->accConstraintSize[tID];

    const graph::Vector LK = l.minus(k);
    const graph::Vector MK = m.minus(k);

    ///
    double value = LK.cross(MK) - param->value * LK.length();
    ///
    mt.setValue(offset, 0, value);
}

///
/// ConstraintAngle2Lines =================================================================
///
__device__ void evaluateValueConstraintAngle2Lines(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    /// coordinate system o_x axis
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);
    const graph::Parameter *param = ec->getParameter(constraint->paramId);
    const int offset = ec->accConstraintSize[tID];

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);

    ///
    double value = LK.product(NM) - LK.length() * NM.length() * cos(toRadians(param->value));
    ///
    mt.setValue(offset, 0, value);
}

///
/// ConstraintSetHorizontal ===============================================================
///
__device__ void evaluateValueConstraintSetHorizontal(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    /// coordinate system oY axis
    static __device__ graph::Vector m = graph::Vector(0.0, 0.0);
    static __device__ graph::Vector n = graph::Vector(0.0, 100.0);

    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const int offset = ec->accConstraintSize[tID];

    ///
    double value = k.minus(l).product(m.minus(n));
    ///
    mt.setValue(offset, 0, value);
}

///
/// ConstraintSetVertical =================================================================
///
__device__ void evaluateValueConstraintSetVertical(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    /// coordinate system oX axis
    static __device__ graph::Vector m = graph::Vector(0.0, 0.0);
    static __device__ graph::Vector n = graph::Vector(100.0, 0.0);
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const int offset = ec->accConstraintSize[tID];

    ///
    double value = k.minus(l).product(m.minus(n));
    ///
    mt.setValue(offset, 0, value);
}

///
/// Evaluate Constraint Value =============================================================
///
__global__ void evaluateConstraintValue(Computation *ec, int N)
{
    int tID;
    graph::Constraint *constraint = nullptr;

    tID = blockDim.x * blockIdx.x + threadIdx.x;

    /// unpack tensor for evaluation
    graph::Tensor mt = graph::Tensor::fromDeviceMem(ec->b + (ec->size), ec->coffSize, 1);

    if (tID < N) {
        constraint = ec->getConstraint(tID);

        switch (constraint->constraintTypeId) {
        case CONSTRAINT_TYPE_ID_FIX_POINT:
            evaluateValueConstraintFixPoint(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_XFIX:
            evaluateValueConstraintParametrizedXfix(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_YFIX:
            evaluateValueConstraintParametrizedYfix(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_CONNECT_2_POINTS:
            evaluateValueConstraintConnect2Points(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_HORIZONTAL_POINT:
            evaluateValueConstraintHorizontalPoint(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_VERTICAL_POINT:
            evaluateValueConstraintVerticalPoint(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PARALLELISM:
            evaluateValueConstraintLinesParallelism(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR:
            evaluateValueConstraintLinesPerpendicular(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_EQUAL_LENGTH:
            evaluateValueConstraintEqualLength(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_LENGTH:
            evaluateValueConstraintParametrizedLength(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_TANGENCY:
            evaluateValueConstrainTangency(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_CIRCLE_TANGENCY:
            evaluateValueConstraintCircleTangency(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_2_POINTS:
            evaluateValueConstraintDistance2Points(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_POINT_LINE:
            evaluateValueConstraintDistancePointLine(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_ANGLE_2_LINES:
            evaluateValueConstraintAngle2Lines(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_SET_HORIZONTAL:
            evaluateValueConstraintSetHorizontal(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_SET_VERTICAL:
            evaluateValueConstraintSetVertical(tID, constraint, ec, mt);
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
__device__ void evaluateJacobianConstraintFixPoint(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const static __device__ graph::Tensor I = graph::SmallTensor::diagonal(1.0);

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    mt.setSubTensor(i * 2, 0, I);    
}

///
/// ConstraintParametrizedXfix    =========================================================
///
__device__ void evaluateJacobianConstraintParametrizedXfix(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    /// wspolrzedna [X]
    mt.setValue(i * 2 + 0, 0 , 1.0);
}

///
/// ConstraintParametrizedYfix    =========================================================
///
__device__ void evaluateJacobianConstraintParametrizedYfix(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    
    int i;

    // k
    i = ec->pointOffset[constraint->k];
    /// wspolrzedna [X]
    mt.setValue(i * 2 + 1, 0, 1.0);

}

///
/// ConstraintConnect2Points    ===========================================================
///
__device__ void evaluateJacobianConstraintConnect2Points(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const static __device__ graph::Tensor I = graph::SmallTensor::diagonal(1.0);
    const static __device__ graph::Tensor mI = graph::SmallTensor::diagonal(-1.0);

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    mt.setSubTensor(i * 2, 0, I); 
    // l
    i = ec->pointOffset[constraint->l];
    mt.setSubTensor(i * 2, 0,  mI);
}

///
/// ConstraintHorizontalPoint    ==========================================================
///
__device__ void evaluateJacobianConstraintHorizontalPoint(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    
    int i;

    i = ec->pointOffset[constraint->k];
    mt.setValue(i * 2, 0, 1.0); // zero-X
    //
    i = ec->pointOffset[constraint->l];
    mt.setValue(i * 2, 0, -1.0); // zero-X
}

///
/// ConstraintVerticalPoint    ============================================================
///
__device__ void evaluateJacobianConstraintVerticalPoint(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
        
    int i;

    i = ec->pointOffset[constraint->k];
    mt.setValue(i * 2, 0, 1.0); // zero-X
    //
    i = ec->pointOffset[constraint->l];
    mt.setValue(i * 2, 0, -1.0); // zero-X
}

///
/// ConstraintLinesParallelism    =========================================================
///
__device__ void evaluateJacobianConstraintLinesParallelism(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);
    
    int i;    
    
    const graph::Vector NM = n.minus(m);
    const graph::Vector LK = l.minus(k);
    
    // k
    i = ec->pointOffset[constraint->k];
    mt.setVector(i * 2, 0, NM.pivot());
    // l
    i = ec->pointOffset[constraint->l];
    mt.setVector(i * 2, 0, NM.pivot().product(-1.0));
    // m
    i = ec->pointOffset[constraint->m];
    mt.setVector(i * 2, 0, LK.pivot().product(-1.0));
    // n
    i = ec->pointOffset[constraint->n];
    mt.setVector(i * 2, 0, LK.pivot());
}

///
/// ConstraintLinesPerpendicular    =======================================================
///
__device__ void evaluateJacobianConstraintLinesPerpendicular(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);

    int i;    
    
    /// K
    i = ec->pointOffset[constraint->k];
    mt.setVector(i * 2, 0, m.minus(n));
    /// L
    i = ec->pointOffset[constraint->l];
    mt.setVector(i * 2, 0, m.minus(n).product(-1.0));
    /// M
    i = ec->pointOffset[constraint->m];
    mt.setVector(i * 2, 0, k.minus(l));
    /// N
    i = ec->pointOffset[constraint->n];
    mt.setVector(i * 2, 0, k.minus(l).product(-1.0));
}

///
/// ConstraintEqualLength    ==============================================================
///
__device__ void evaluateJacobianConstraintEqualLength(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);

    const graph::Vector vLK = l.minus(k).unit();
    const graph::Vector vNM = n.minus(m).unit();

    int i;

    //k
    i = ec->pointOffset[constraint->k];
    mt.setVector(i * 2, 0, vLK.product(-1.0));
    //l
    i = ec->pointOffset[constraint->l];
    mt.setVector(i * 2, 0, vLK);
    //m
    i = ec->pointOffset[constraint->m];
    mt.setVector(i * 2, 0, vNM);
    //n
    i = ec->pointOffset[constraint->n];
    mt.setVector(i * 2, 0, vNM.product(-1.0));
}

///
/// ConstraintParametrizedLength    =======================================================
///
__device__ void evaluateJacobianConstraintParametrizedLength(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);

    const double lk = LK.length();
    const double nm = NM.length();

    const double d = ec->getParameter(constraint->paramId)->value;

    int i;

    //k
    i = ec->pointOffset[constraint->k];
    mt.setVector(i * 2, 0, LK.product(-1.0 * d / lk));
    //l
    i = ec->pointOffset[constraint->l];
    mt.setVector(i * 2, 0, LK.product(1.0 * d / lk));
    //m
    i = ec->pointOffset[constraint->m];
    mt.setVector(i * 2, 0, NM.product(1.0 / nm));
    //n
    i = ec->pointOffset[constraint->n];
    mt.setVector(i * 2, 0, NM.product(-1.0 / nm));

}

///
/// ConstrainTangency    ==================================================================
///
__device__ void evaluateJacobianConstrainTangency(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);

    const graph::Vector MK = m.minus(k);
    const graph::Vector LK = l.minus(k);
    const graph::Vector ML = m.minus(l);
    const graph::Vector NM = n.minus(m);
    const double nm = NM.product(NM);
    const double lk = LK.product(LK);
    const double CRS = LK.cross(MK);

    int i;

    //k
    i = ec->pointOffset[constraint->k];
    mt.setVector(i * 2, 0, ML.pivot().product(2.0 * CRS).plus(LK.product(2.0 * nm)));
    //l
    i = ec->pointOffset[constraint->l];
    mt.setVector(i * 2, 0, MK.pivot().product(-2.0 * CRS).plus(LK.product(-2.0 * nm)));
    //m
    i = ec->pointOffset[constraint->m];
    mt.setVector(i * 2, 0, LK.pivot().product(2.0 * CRS).plus(NM.product(2.0 * lk)));
    //n
    i = ec->pointOffset[constraint->n];
    mt.setVector(i * 2, 0, NM.product(-2.0 * lk));
}

///
/// ConstraintCircleTangency    ===========================================================
///
__device__ void evaluateJacobianConstraintCircleTangency(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);

    const graph::Vector vLK = l.minus(k).unit();
    const graph::Vector vNM = n.minus(m).unit();
    const graph::Vector vMK = m.minus(k).unit();

    int i;

    //k
    i = ec->pointOffset[constraint->k];
    mt.setVector(i * 2, 0, vMK.minus(vLK));
    //l
    i = ec->pointOffset[constraint->l];
    mt.setVector(i * 2, 0, vLK);
    //m
    i = ec->pointOffset[constraint->m];
    mt.setVector(i * 2, 0, vMK.product(-1.0).minus(vNM));
    //n
    i = ec->pointOffset[constraint->n];
    mt.setVector(i * 2, 0, vNM);
}

///
/// ConstraintDistance2Points    ==========================================================
///
__device__ void evaluateJacobianConstraintDistance2Points(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);

    const graph::Vector vLK = l.minus(k).unit();

    int i;

    //k
    i = ec->pointOffset[constraint->k];
    mt.setVector( i * 2, 0, vLK.product(-1.0));
    //l
    i = i = ec->pointOffset[constraint->l];
    mt.setVector( i * 2, 0, vLK);
}

///
/// ConstraintDistancePointLine    ========================================================
///
__device__ void evaluateJacobianConstraintDistancePointLine(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);

    const graph::Vector LK = l.minus(k);
    const graph::Vector MK = m.minus(k);

    const double d = ec->getParameter(constraint->paramId)->value;

    int i;

    //k
    i = ec->pointOffset[constraint->k];
    mt.setVector( 2 * i, 0, LK.product(d / LK.length()).minus(MK.plus(LK).pivot()));
    //l
    i = ec->pointOffset[constraint->l];
    mt.setVector( 2 * i, 0, LK.product( -1.0 * d / LK.length()));
    //m
    i = ec->pointOffset[constraint->m];
    mt.setVector( 2 * i, 0, LK.pivot());
}

///
/// ConstraintAngle2Lines    ==============================================================
///
__device__ void evaluateJacobianConstraintAngle2Lines(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);

    const double d = ec->getParameter(constraint->paramId)->value;
    const double rad = toRadians(d);

    const graph::Vector LK = l.minus(k);
    const graph::Vector NM = n.minus(m);
    const graph::Vector uLKdNM = LK.unit().product(NM.length()).product(cos(rad));
    const graph::Vector uNMdLK = NM.unit().product(LK.length()).product(cos(rad));

    int i;

    //k
    i = ec->pointOffset[constraint->k];
    mt.setVector(i * 2, 0, uLKdNM.minus(NM));
    //l
    i = ec->pointOffset[constraint->l];
    mt.setVector(i * 2, 0, NM.minus(uLKdNM));
    //m
    i = ec->pointOffset[constraint->m];
    mt.setVector(i * 2, 0, uNMdLK.minus(LK));
    //n
    i = ec->pointOffset[constraint->n];
    mt.setVector(i * 2, 0, LK.minus(uNMdLK));
}

///
/// ConstraintSetHorizontal    ============================================================
///
__device__ void evaluateJacobianConstraintSetHorizontal(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    static __device__ graph::Vector m = graph::Vector(0.0, 0.0);
    static __device__ graph::Vector n = graph::Vector(0.0, 100.0);

    int i;

    /// K
    i = ec->pointOffset[constraint->k];
    mt.setVector(i * 2, 0, m.minus(n));
    /// L
    i = ec->pointOffset[constraint->l];
    mt.setVector(i * 2, 0, m.minus(n).product(-1.0));
}

///
/// ConstraintSetVertical    ==============================================================
///
__device__ void evaluateJacobianConstraintSetVertical(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    static __device__ graph::Vector m = graph::Vector(0.0, 0.0);
    static __device__ graph::Vector n = graph::Vector(100.0, 0.0);

    int i;

    /// K
    i = ec->pointOffset[constraint->k];
    mt.setVector(i * 2, 0, m.minus(n));
    /// L
    i = ec->pointOffset[constraint->l];
    mt.setVector(i * 2, 0, m.minus(n).product(-1.0));
}

///
/// Evaluate Constraint Jacobian ==========================================================
/// 
/// 
/// (FI)' - (dfi/dq)` transponowane - upper traingular matrix A
///
/// 
__global__ void evaluateConstraintJacobian(Computation *ec, int N)
{

    int tID;
    graph::Constraint *constraint = nullptr;

    tID = blockDim.x * blockIdx.x + threadIdx.x;

    /// unpack tensor for evaluation
    /// COLUMN_ORDER - tensor A
    int constraintOffset = ec->accConstraintSize[tID] + (ec->size);
    graph::Tensor mt = graph::Tensor::fromDeviceMem(ec->A + (ec->dimension) * constraintOffset, ec->size, 1);

    if (tID < N) {

        constraint = ec->getConstraint(tID);

        switch (constraint->constraintTypeId) {
        case CONSTRAINT_TYPE_ID_FIX_POINT:
            evaluateJacobianConstraintFixPoint(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_XFIX:
            evaluateJacobianConstraintParametrizedXfix(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_YFIX:
            evaluateJacobianConstraintParametrizedYfix(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_CONNECT_2_POINTS:
            evaluateJacobianConstraintConnect2Points(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_HORIZONTAL_POINT:
            evaluateJacobianConstraintHorizontalPoint(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_VERTICAL_POINT:
            evaluateJacobianConstraintVerticalPoint(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PARALLELISM:
            evaluateJacobianConstraintLinesParallelism(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR:
            evaluateJacobianConstraintLinesPerpendicular(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_EQUAL_LENGTH:
            evaluateJacobianConstraintEqualLength(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_LENGTH:
            evaluateJacobianConstraintParametrizedLength(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_TANGENCY:
            evaluateJacobianConstrainTangency(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_CIRCLE_TANGENCY:
            evaluateJacobianConstraintCircleTangency(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_2_POINTS:
            evaluateJacobianConstraintDistance2Points(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_POINT_LINE:
            evaluateJacobianConstraintDistancePointLine(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_ANGLE_2_LINES:
            evaluateJacobianConstraintAngle2Lines(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_SET_HORIZONTAL:
            evaluateJacobianConstraintSetHorizontal(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_SET_VERTICAL:
            evaluateJacobianConstraintSetVertical(tID, constraint, ec, mt);
            break;
        default:
            break;
        }
        return;
    }
}


#endif // #ifndef _SOLVER_KERNEL_CUH_