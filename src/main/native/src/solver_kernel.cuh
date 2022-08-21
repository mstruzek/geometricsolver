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

///         allocate in `setup` function  , all references are valid GPU memory references
///
///         cudaMalloc, cudaMemcpy
///
// - 
struct Computation {
// immutable
    int computationId;
    int norm2;          ///  cublasDnrm2(...)
    int info;           /// device variable cuBlas


    int size;           /// wektor stanu
    int coffSize;       /// wspolczynniki Lagrange
    int dimension;      /// N - dimension = size + coffSize


// macierze ukladu rowan - this computation only
    double *A;  
    double *SV;         /// State Vector  [ SV = SV + dx ] , previous task -- "lineage"
    double *dx;         /// przyrosty   [ A * dx = b ]
    double *b;


// immutable mainly static data
    graph::Point *points;
    graph::Geometric *geometrics;
    graph::Constraint *constraints;
    graph::Parameter *parameters;

    const int *pointOffset;             /// ---
    const int *parameterOffset;         /// paramater offset from given ID
    const int *accGeometricSize;        /// accumulative offset with geometric size evaluation function
    const int *accConstraintSize;       /// accumulative offset with constraint size evaluation function

    /// w drugiej fazie dopisac => dodatkowo potrzebny per block  __shared__  memory   ORR    L1   fast region memory

    __host__ __device__ graph::Point const &getPoint(int pointId) const;

    __host__ __device__ graph::Vector const &Computation::getPointSV(int pointId) const;

    __host__ __device__ graph::Geometric *getGeomtricObject(int geometricId) const;

    __host__ __device__ graph::Constraint *getConstraint(int constraintId) const;

    __host__ __device__ graph::Parameter *getParameter(int parameterId) const;

    __host__ __device__ double getLagrangeMultiplier(int constraintId) const;
};

__host__ __device__ graph::Point const &Computation::getPoint(int pointId) const {
    int offset = pointOffset[pointId];
    return points[offset];
}

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

__host__ __device__ double toRadians(double value) { 
    return (M_PI / 180.0) * value; 
}

__host__ __device__ double Computation::getLagrangeMultiplier(int constraintId) const {
    int multiOffset = accConstraintSize[constraintId];
    return SV[size + multiOffset];
}

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
__global__ void ComputeStiffnessMatrix(Computation *ec, int N) {
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

__device__ void setForceIntensity_FreePoint(int tID, graph::Geometric *geometric, Computation *ec, graph::Tensor &mt) {
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

__device__ void setForceIntensity_Line(int tID, graph::Geometric *geometric, Computation *ec, graph::Tensor &mt) {
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

__device__ void setForceIntensity_FixLine(int tID, graph::Geometric *geometric, Computation *ec, graph::Tensor &mt) {}

///
/// Circle  ===============================================================================
///

__device__ void setForceIntensity_Circle(int tID, graph::Geometric *geometric, Computation *ec, graph::Tensor &mt) {
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

__device__ void setForceIntensity_Arc(int tID, graph::Geometric *geometric, Computation *ec, graph::Tensor &mt) {
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

__global__ void EvaluateForceIntensity(Computation *ec, int N) {
    int tID;
    graph::Geometric *geometric = nullptr;

    tID = blockDim.x * blockIdx.x + threadIdx.x;

    /// unpack tensor for evaluation
    graph::Tensor mt = graph::Tensor::fromDeviceMem(ec->b, ec->size, 1);

    if (tID < N) {
        geometric = ec->getGeomtricObject(tID);

        switch (geometric->geometricTypeId) {

        case GEOMETRIC_TYPE_ID_FREE_POINT:
            setForceIntensity_FreePoint(tID, geometric, ec, mt);
            break;

        case GEOMETRIC_TYPE_ID_LINE:
            setForceIntensity_Line(tID, geometric, ec, mt);
            break;

        case GEOMETRIC_TYPE_ID_FIX_LINE:
            setForceIntensity_FixLine(tID, geometric, ec, mt);
            break;

        case GEOMETRIC_TYPE_ID_CIRCLE:
            setForceIntensity_Circle(tID, geometric, ec, mt);
            break;

        case GEOMETRIC_TYPE_ID_ARC:
            setForceIntensity_Arc(tID, geometric, ec, mt);
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
__device__ void setValueConstraintFixPoint(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
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
__device__ void setValueConstraintParametrizedXfix(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
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
__device__ void setValueConstraintParametrizedYfix(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
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
__device__ void setValueConstraintConnect2Points(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
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
__device__ void setValueConstraintHorizontalPoint(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
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
__device__ void setValueConstraintVerticalPoint(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
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
__device__ void setValueConstraintLinesParallelism(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
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
__device__ void setValueConstraintLinesPerpendicular(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
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
__device__ void setValueConstraintEqualLength(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
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
__device__ void setValueConstraintParametrizedLength(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
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
__device__ void setValueConstrainTangency(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
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
__device__ void setValueConstraintCircleTangency(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
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
__device__ void setValueConstraintDistance2Points(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
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
__device__ void setValueConstraintDistancePointLine(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
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
__device__ void setValueConstraintAngle2Lines(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
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
__device__ void setValueConstraintSetHorizontal(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    /// coordinate system oY axis
    const graph::Vector m = graph::Vector(0.0, 0.0);
    const graph::Vector n = graph::Vector(0.0, 100.0);

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
__device__ void setValueConstraintSetVertical(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt) {
    /// coordinate system oX axis
    const graph::Vector m = graph::Vector(0.0, 0.0);
    const graph::Vector n = graph::Vector(100.0, 0.0);
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
__global__ void EvaluateConstraintValue(Computation *ec, int N)
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
            setValueConstraintFixPoint(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_XFIX:
            setValueConstraintParametrizedXfix(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_YFIX:
            setValueConstraintParametrizedYfix(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_CONNECT_2_POINTS:
            setValueConstraintConnect2Points(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_HORIZONTAL_POINT:
            setValueConstraintHorizontalPoint(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_VERTICAL_POINT:
            setValueConstraintVerticalPoint(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PARALLELISM:
            setValueConstraintLinesParallelism(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR:
            setValueConstraintLinesPerpendicular(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_EQUAL_LENGTH:
            setValueConstraintEqualLength(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_PARAMETRIZED_LENGTH:
            setValueConstraintParametrizedLength(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_TANGENCY:
            setValueConstrainTangency(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_CIRCLE_TANGENCY:
            setValueConstraintCircleTangency(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_2_POINTS:
            setValueConstraintDistance2Points(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_DISTANCE_POINT_LINE:
            setValueConstraintDistancePointLine(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_ANGLE_2_LINES:
            setValueConstraintAngle2Lines(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_SET_HORIZONTAL:
            setValueConstraintSetHorizontal(tID, constraint, ec, mt);
            break;
        case CONSTRAINT_TYPE_ID_SET_VERTICAL:
            setValueConstraintSetVertical(tID, constraint, ec, mt);
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
__device__ void setJacobianConstraintFixPoint(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const graph::Tensor I = graph::SmallTensor::diagonal(1.0);

    int i;

    // k
    i = ec->pointOffset[constraint->k];
    mt.setSubTensor(i * 2, 0, I);    
}

///
/// ConstraintParametrizedXfix    =========================================================
///
__device__ void setJacobianConstraintParametrizedXfix(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
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
__device__ void setJacobianConstraintParametrizedYfix(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
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
__device__ void setJacobianConstraintConnect2Points(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const graph::Tensor I = graph::SmallTensor::diagonal(1.0);
    const graph::Tensor mI = graph::SmallTensor::diagonal(-1.0);

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
__device__ void setJacobianConstraintHorizontalPoint(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
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
__device__ void setJacobianConstraintVerticalPoint(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
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
__device__ void setJacobianConstraintLinesParallelism(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
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
__device__ void setJacobianConstraintLinesPerpendicular(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
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
__device__ void setJacobianConstraintEqualLength(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
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
__device__ void setJacobianConstraintParametrizedLength(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
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
__device__ void setJacobianConstrainTangency(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
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
__device__ void setJacobianConstraintCircleTangency(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
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
__device__ void setJacobianConstraintDistance2Points(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
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
__device__ void setJacobianConstraintDistancePointLine(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
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
__device__ void setJacobianConstraintAngle2Lines(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
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
__device__ void setJacobianConstraintSetHorizontal(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const graph::Vector m = graph::Vector(0.0, 0.0);
    const graph::Vector n = graph::Vector(0.0, 100.0);

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
__device__ void setJacobianConstraintSetVertical(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const graph::Vector m = graph::Vector(0.0, 0.0);
    const graph::Vector n = graph::Vector(100.0, 0.0);

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
__global__ void EvaluateConstraintJacobian(Computation *ec, int N)
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
__device__ void setHessianTensorConstraintFixPoint(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintParametrizedXfix  ===========================================================
///
__device__ void setHessianTensorConstraintParametrizedXfix(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintParametrizedYfix  ===========================================================
///
__device__ void setHessianTensorConstraintParametrizedYfix(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintConnect2Points  =============================================================
///
__device__ void setHessianTensorConstraintConnect2Points(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintHorizontalPoint  ============================================================
///
__device__ void setHessianTensorConstraintHorizontalPoint(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintVerticalPoint  ==============================================================
///
__device__ void setHessianTensorConstraintVerticalPoint(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintLinesParallelism  ===========================================================
///
__device__ void setHessianTensorConstraintLinesParallelism(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
        /// macierz NxN
        const double lagrange = ec->getLagrangeMultiplier(tID);
        const graph::Tensor R = graph::SmallTensor::rotation(90 + 180).multiplyC(lagrange);     /// R
        const graph::Tensor Rm = graph::SmallTensor::rotation(90).multiplyC(lagrange);          /// Rm = -R

        int i;
        int j;


        //k,m
        i = ec->pointOffset[constraint->k];
        j = ec->pointOffset[constraint->m];
        mt.plusSubTensor(2 * i, 2 * j, R);

        //k,n
        i = ec->pointOffset[constraint->k];
        j = ec->pointOffset[constraint->n];
        mt.plusSubTensor(2 * i, 2 * j, Rm);

        //l,m
        i = ec->pointOffset[constraint->l];
        j = ec->pointOffset[constraint->m];
        mt.plusSubTensor(2 * i, 2 * j, Rm);

        //l,n
        i = ec->pointOffset[constraint->l];
        j = ec->pointOffset[constraint->n];
        mt.plusSubTensor(2 * i, 2 * j, R);

        //m,k
        i = ec->pointOffset[constraint->m];
        j = ec->pointOffset[constraint->k];
        mt.plusSubTensor(2 * i, 2 * j, Rm);

        //m,l
        i = ec->pointOffset[constraint->m];
        j = ec->pointOffset[constraint->l];
        mt.plusSubTensor(2 * i, 2 * j, R);

        //n,k
        i = ec->pointOffset[constraint->n];
        j = ec->pointOffset[constraint->k];
        mt.plusSubTensor(2 * i, 2 * j, R);

        //n,l
        i = ec->pointOffset[constraint->n];
        j = ec->pointOffset[constraint->l];
        mt.plusSubTensor(2 * i, 2 * j, Rm);

}

///
/// ConstraintLinesPerpendicular  =========================================================
///
__device__ void setHessianTensorConstraintLinesPerpendicular(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
        const double lagrange = ec->getLagrangeMultiplier(tID);
        const graph::Tensor I = graph::SmallTensor::identity(1.0 * lagrange);
        const graph::Tensor Im = graph::SmallTensor::identity(1.0 * lagrange);

        int i;
        int j;

        //wstawiamy I,-I w odpowiednie miejsca
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
__device__ void setHessianTensorConstraintEqualLength(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintParametrizedLength  =========================================================
///
__device__ void setHessianTensorConstraintParametrizedLength(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstrainTangency  ====================================================================
///
__device__ void setHessianTensorConstrainTangency(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    /// equation error - java impl
}

///
/// ConstraintCircleTangency  =============================================================
///
__device__ void setHessianTensorConstraintCircleTangency(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    /// no equation from - java impl
}

///
/// ConstraintDistance2Points  ============================================================
///
__device__ void setHessianTensorConstraintDistance2Points(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintDistancePointLine  ==========================================================
///
__device__ void setHessianTensorConstraintDistancePointLine(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    /// equation error - java impl
}

///
/// ConstraintAngle2Lines  ================================================================
///
__device__ void setHessianTensorConstraintAngle2Lines(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    const graph::Point &k = ec->getPoint(constraint->k);
    const graph::Point &l = ec->getPoint(constraint->l);
    const graph::Point &m = ec->getPoint(constraint->m);
    const graph::Point &n = ec->getPoint(constraint->n);

    const double lagrange = ec->getLagrangeMultiplier(tID);

    const double d = ec->getParameter(constraint->paramId)->value;
    const double rad = toRadians(d);

    const graph::Vector LK = l.minus(k).unit();
    const graph::Vector NM = n.minus(m).unit();
    double g = LK.product(NM) * cos(rad);

    const graph::Tensor I_1G = graph::SmallTensor::diagonal((1 - g) * lagrange);
    const graph::Tensor I_Gd  = graph::SmallTensor::diagonal((g - 1) * lagrange);

    int i;
    int j;

    //k,k
    i = ec->pointOffset[constraint->k];
    j = ec->pointOffset[constraint->k];
    // 0

    //k,l
    i = ec->pointOffset[constraint->k];
    j = ec->pointOffset[constraint->l];
    //0

    //k,m
    i = ec->pointOffset[constraint->k];
    j = ec->pointOffset[constraint->m];
    mt.plusSubTensor(2 * i, 2 * j, I_1G);

    //k,n
    i = ec->pointOffset[constraint->k];
    j = ec->pointOffset[constraint->n];
    mt.plusSubTensor(2 * i, 2 * j, I_Gd);

    //l,k
    i = ec->pointOffset[constraint->l];
    j = ec->pointOffset[constraint->k];
    //0

    //l,l
    i = ec->pointOffset[constraint->l];
    j = ec->pointOffset[constraint->l];
    // 0

    //l,m
    i = ec->pointOffset[constraint->l];
    j = ec->pointOffset[constraint->m];
    mt.plusSubTensor(2 * i, 2 * j, I_Gd);

    //l,n
    i = ec->pointOffset[constraint->l];
    j = ec->pointOffset[constraint->n];
    mt.plusSubTensor(2 * i, 2 * j, I_1G);

    //m,k
    i = ec->pointOffset[constraint->m];
    j = ec->pointOffset[constraint->k];
    mt.plusSubTensor(2 * i, 2 * j, I_1G);

    //m,l
    i = ec->pointOffset[constraint->m];
    j = ec->pointOffset[constraint->l];
    mt.plusSubTensor(2 * i, 2 * j, I_Gd);

    //m,m
    i = ec->pointOffset[constraint->m];
    j = ec->pointOffset[constraint->m];
    //0

    //m,n
    i = ec->pointOffset[constraint->m];
    j = ec->pointOffset[constraint->n];
    //0

    //n,k
    i = ec->pointOffset[constraint->n];
    j = ec->pointOffset[constraint->k];
    mt.plusSubTensor(2 * i, 2 * j, I_Gd);

    //n,l
    i = ec->pointOffset[constraint->n];
    j = ec->pointOffset[constraint->l];
    mt.plusSubTensor(2 * i, 2 * j, I_1G);

    //n,m
    i = ec->pointOffset[constraint->n];
    j = ec->pointOffset[constraint->m];
    //0

    //n,n
    i = ec->pointOffset[constraint->n];
    j = ec->pointOffset[constraint->n];
    //0
}

///
/// ConstraintSetHorizontal  ==============================================================
///
__device__ void setHessianTensorConstraintSetHorizontal(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
{
    /// empty
}

///
/// ConstraintSetVertical  ================================================================
///
__device__ void setHessianTensorConstraintSetVertical(int tID, graph::Constraint *constraint, Computation *ec, graph::Tensor &mt)
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
__global__ void EvaluateConstraintHessian(Computation *ec, int N)
{
    int tID;
    graph::Constraint *constraint = nullptr;

    tID = blockDim.x * blockIdx.x + threadIdx.x;

    /// unpack tensor for evaluation
    /// COLUMN_ORDER - tensor A

    graph::Tensor mt = graph::Tensor::fromDeviceMem(ec->A, ec->size, 1);

    if (tID < N) {

        constraint = ec->getConstraint(tID);

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
#endif // #ifndef _SOLVER_KERNEL_CUH_