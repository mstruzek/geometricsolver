#include "geometric_solver.h"

#include <memory>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "stop_watch.h"


#include "model.cuh"
#include "model_config.h"



void checkCudaStatus_impl(cudaError_t status, size_t __line_) {
        if (status != cudaSuccess) {
                printf("%li: cuda API failed with status %d\n", __line_, status);
                throw std::logic_error("cuda API error");
        }
}


///
/// Setup all matricies for computation and prepare kernel stream  intertwined with cuSolver
///
///
void solveSystemOnGPU(
    std::vector<graph::Point> const& points, 
    std::vector<graph::Geometric> const& geometrics,
    std::vector<graph::Constraint> const& constraints,
    std::vector<graph::Parameter> const& parameters, 
    std::shared_ptr<int[]> pointOffset, 
    std::shared_ptr<int[]> constraintOffset,
    std::shared_ptr<int[]> geometricOffset, 
    graph::SolverStat *stat, 
    int *err) 
{

        constexpr int N = 1024;

        constexpr int size = N * sizeof(int);

        int* a;
        int* b;
        int* c;

        int* HA;
        int* HB;
        int* HC;

        checkCudaStatus(cudaSetDevice(0));

/// HOST#
        a = (int*) malloc(size);
        b = (int*) malloc(size);
        c = (int*) malloc(size);

        printf("mem allocated \n");

/// DEVICE#
        checkCudaStatus(cudaMalloc((void**)&HA, size));
        checkCudaStatus(cudaMalloc((void**)&HB, size));
        checkCudaStatus(cudaMalloc((void**)&HC, size));

        printf("cuda malloc \n");

/// --- test data
        for(int i = 0 ; i < N ; i++) {
                a[i] = i;
                b[i] = N-i;
                c[i] = 0;
        }

        checkCudaStatus(cudaMemcpy((void*)HA, a, size, cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy((void*)HB, b, size, cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy((void*)HC, c, size, cudaMemcpyHostToDevice));

        printf("cuda memcpy \n");

        int gridSize = 1;
        int blockSize = 1024;

        kernel_add<<<gridSize, blockSize>>>(HA,HB,HC, N);


        checkCudaStatus(cudaMemcpy((void*)c, HC, size, cudaMemcpyDeviceToHost));

        for(int i= 0; i < N ; i++) {
                printf("%d  --- %d \n",i , c[i]);
        }

        checkCudaStatus(cudaFree(HA));
        checkCudaStatus(cudaFree(HB));
        checkCudaStatus(cudaFree(HC));

        free(a);
        free(b);
        free(c);

        *error = 0;      
}



/// Accumulated Constraint Size
static std::unique_ptr<int[]> accConstraintSize;

/// Accymulated Geometric Object Size
static std::unique_ptr<int[]> accGeometricSize; /// 2 * point.size()


static graph::StopWatch solverWatch;
static graph::StopWatch accEvoWatch;
static graph::StopWatch accLUWatch;

/// CPU#
long AllLagrangeCoffSize() { return 0; }

/// KERNEL#
double ConstraintGetFullNorm() { return 0.0; }

/// CPU#
void PointLocationSetup() {}

/// CPUtoGPU#
void CopyIntoStateVector(void *) {}

/// CPU# and GPU#
void SetupLagrangeMultipliers(void) {}

/// KERNEL#
void GeometricObjectEvaluateForceVector() { /// Sily  - F(q)
        // b.mulitply(-1);
}

// KERNEL#
void ConstraintEvaluateConstraintVector() {
        /// Wiezy  - Fi(q)

        /// b.mulitply(-1);
}

// KERNEL#
void ConstraintGetFullJacobian() {}

/// KERNEL# and last step update CPU memory for JNI synchronizationa
void PointUtilityCopyFromStateVector() {}

/// KERNEL#
void ConstraintGetFullHessian() {}

#define MAX_SOLVER_ITERATIONS 20

#define CONVERGENCE_LIMIT 10e-5

template <typename Obj, typename ObjIdFunction> std::unique_ptr<int[]> stateOffset(std::vector<Obj> objects, ObjIdFunction objectIdFunction) {
        std::unique_ptr<int[]> offsets(new int[objectIdFunction(objects.rbegin())]);
        auto iterator = objects.begin();
        int offset = 0;
        while (iterator != objects.end()) {
                offsets[offset++] = objectIdFunction(iterator);
                iterator++;
        }
        return offsets;
}

/// #/include <numeric> std::partial_sum

template <typename Obj, typename ValueFunction> std::unique_ptr<int[]> accumalatedValue(std::vector<Obj> vector, ValueFunction valueFunction) {
        int accValue = 0;
        std::unique_ptr<int[]> accumulated(new int[vector.size()]);
        for (int offset = 0; offset < vector.size(); offset++) {
                accumulated[offset] = accValue;
                accValue = accValue + valueFunction(vector[offset]);
        }
        return accumulated;
}



/// KERNEL# GPU
__global__ void kernel_add(int* HA, int* HB, int* HC, int size) 
{

        int i = threadIdx.x;

        if(i < size) {

                HC[i] = HA[i] + HB[i];
        }

}


/// DEVICE# context


/// KERNEL# 
///         --- allocate in `setup` function  , all references are valid GPU memory references
///
///         cudaMalloc, cudaMemcpy
///
struct EvaluationContext {

    graph::Point* const points;         

    /// w drugiej fazie dopisac => dodatkowo potrzebny per block  __shared__  memory   ORR    L1   fast region memory
    
    const graph::Geometric* const geometrics;
    const graph::Constraint* const constraints;
    const graph::Parameter* const parameters;    

    const int* const pointOffset;           /// ---
    const int* const constraintOffset;      /// accumulative offset with constraint size evaluation function     
    const int* const geometricOffset;       /// accumulative offset with geometric size evaluation function
    
    graph::SolverStat* const stat;

    __device__ graph::Point const& getPoint(int pointId) const;

    __device__ graph::Geometric *  getGeomtricObject(int geometricId) const;
};

__device__ 
graph::Point const& EvaluationContext::getPoint(int pointId) const
{
        int offset = pointOffset[pointId];
        return points[offset];
};

__device__ 
graph::Geometric*  EvaluationContext::getGeomtricObject(int geometricId) const 
{
        return geometrics[geometricId];
};


/// @brief Setup __device__ vector of geometric points in this moment.
/// 
/// @param ec evaluation context
/// @param N size of geometric object data vector
/// @param _point[] __shared__ reference into model point state  
/// @tparam TZ tensor dimension without constraints 
/// @return void
///
template<size_t TZ>
__device__ void SetupPointAccessor( EvaluationContext* ec , int N, /* __shared__*/ double _points[TZ]) 
{    
    int offset;    
    int tID = blockDim.x * blockIdx.x  + threadIdx.x;

    /// set-aside - persisten access in L2 cache !    
    /// more than 1024 points will not share this reference
    
   
    graph::Tensor tensor; // view into Evaluation Context reference

    if(tID < N ) 
    {
        graph::Geometric *geometric = ec->getGeomtricObject(tID);

        offset = tID + ec->geometricOffset[tID];

        switch(geometric->geometricTypeId) {

            case GEOMETRIC_TYPE_ID_FREE_POINT: 

                int aID = geometric->a;
                int bID = geometric->b;
                int p1ID = geometric->p1;

                graph::Point const& a   = ec->getPoint(aID);
                graph::Point const& b   = ec->getPoint(bID);
                graph::Point const& p1  = ec->getPoint(p1ID);
        /// ======================================================/// 
                _points[offset + 0] =  a.x;
                _points[offset + 1] =  a.y;
                
                _points[offset + 2] =  p1.x;
                _points[offset + 3] =  p1.y;

                _points[offset + 4] =  b.x;
                _points[offset + 5] =  b.y;
        /// ======================================================/// 
                break;

            case GEOMETRIC_TYPE_ID_LINE: 
                int aID = geometric->a;
                int bID = geometric->b;
                int p1ID = geometric->p1;
                int p2ID = geometric->p2;

                graph::Point const& a   = ec->getPoint(aID);
                graph::Point const& b   = ec->getPoint(bID);
                graph::Point const& p1  = ec->getPoint(p1ID);
                graph::Point const& p2  = ec->getPoint(p2ID);
        /// ======================================================/// 
                _points[offset + 0] =  a.x;
                _points[offset + 1] =  a.y;
                
                _points[offset + 2] =  p1.x;
                _points[offset + 3] =  p1.y;

                _points[offset + 4] =  p2.x;
                _points[offset + 5] =  p2.y;

                _points[offset + 6] =  b.x;
                _points[offset + 7] =  b.y;
        /// ======================================================/// 
                break;

            case GEOMETRIC_TYPE_ID_FIX_LINE: 
            
        /// ======================================================/// 
                break;

            case GEOMETRIC_TYPE_ID_CIRCLE: 
                int aID = geometric->a;
                int bID = geometric->b;
                int p1ID = geometric->p1;
                int p2ID = geometric->p2;

                graph::Point const& a   = ec->getPoint(aID);
                graph::Point const& b   = ec->getPoint(bID);
                graph::Point const& p1  = ec->getPoint(p1ID);
                graph::Point const& p2  = ec->getPoint(p2ID);
        /// ======================================================/// 
                _points[offset + 0] =  a.x;
                _points[offset + 1] =  a.y;
                
                _points[offset + 2] =  p1.x;
                _points[offset + 3] =  p1.y;

                _points[offset + 4] =  p2.x;
                _points[offset + 5] =  p2.y;

                _points[offset + 6] =  b.x;
                _points[offset + 7] =  b.y;
        /// ======================================================/// 
                break;

            case GEOMETRIC_TYPE_ID_ARC: 
                int aID = geometric->a;
                int bID = geometric->b;
                int cID = geometric->c;
                int dID = geometric->d;
                int p1ID = geometric->p1;
                int p2ID = geometric->p2;
                int p3ID = geometric->p3;

                graph::Point const& a   = ec->getPoint(aID);
                graph::Point const& b   = ec->getPoint(bID);
                graph::Point const& c   = ec->getPoint(cID);
                graph::Point const& d   = ec->getPoint(dID);
                graph::Point const& p1  = ec->getPoint(p1ID);
                graph::Point const& p2  = ec->getPoint(p2ID);
                graph::Point const& p3  = ec->getPoint(p3ID);
        /// ======================================================/// 
                _points[offset + 0] =  a.x;
                _points[offset + 1] =  a.y;
                
                _points[offset + 2] =  b.x;
                _points[offset + 3] =  b.y;

                _points[offset + 4] =  c.x;
                _points[offset + 5] =  c.y;

                _points[offset + 6] =  d.x;
                _points[offset + 7] =  d.y;

                _points[offset + 8] =  p1.x;
                _points[offset + 9] =  p1.y;

                _points[offset + 10] =  p2.x;
                _points[offset + 11] =  p2.y;

                _points[offset + 12] =  p3.x;
                _points[offset + 13] =  p3.y;
        /// ======================================================/// 
                break;

            default:
                printf("[error] type of geometric not recognized \n");
                break;
        }
    }
}


__device__ void setStiffnessMatrix_FreePoint(int tid, graph::Tensor& mt);

__device__ void setStiffnessMatrix_Line     (int tid, graph::Tensor& mt);

__device__ void setStiffnessMatrix_FixLine  (int tid, graph::Tensor& mt);

__device__ void setStiffnessMatrix_Circle   (int tid, graph::Tensor& mt);

__device__ void setStiffnessMatrix_Arc      (int tid, graph::Tensor& mt);


/**
 * @brief Compute Stiffness Matrix on each geometric object.
 * 
 * Single cuda thread is responsible for evalution of an assigned geometric object.
 * 
 * 
 * @param ec 
 * @return __global__ 
 */
__global__ void computeStiffnessMatrix( EvaluationContext* ec, int N)
{
    int tID = blockDim.x * blockIdx.x  + threadIdx.x;

    graph::Tensor tensor; // view into Evaluation Context reference

    if( tID < N ) 
    {
        graph::Geometric *geometric = ec->getGeomtricObject(tID);

        switch(geometric->geometricTypeId) {

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
                printf("[error] type of geometric not recognized \n");
                break;

        }
    }
    return;    
}


__device__ void evaluateForceIntensity_FreePoint(int tID, graph::Geometric* geometric, EvaluationContext* ec, graph::Tensor& mt);

__device__ void evaluateForceIntensity_Line     (int tID, graph::Geometric* geometric, EvaluationContext* ec, graph::Tensor& mt);

__device__ void evaluateForceIntensity_FixLine  (int tID, graph::Geometric* geometric, EvaluationContext* ec, graph::Tensor& mt);

__device__ void evaluateForceIntensity_Circle   (int tID, graph::Geometric* geometric, EvaluationContext* ec, graph::Tensor& mt);

__device__ void evaluateForceIntensity_Arc      (int tID, graph::Geometric* geometric, EvaluationContext* ec, graph::Tensor& mt);



__global__ void evaluateForceIntensity( EvaluationContext* ec, int N ) 
{
    int tID;
    graph::Geometric *geometric = nullptr;
    graph::Tensor mt; // VECTOR   :: view into Evaluation Context reference

    
    tID = blockDim.x * blockIdx.x  + threadIdx.x;   
    
     
    if( tID < N) 
    {
        geometric = ec->getGeomtricObject(tID);   

        switch(geometric->geometricTypeId) {

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
                printf("[error] type of geometric not recognized \n");
                break;
        }
    }

    return;    
}


/// functional style - all evaluations in single one function block


///
/// ================================ FORCE INTENSITY ==================== ///
///

/// 1# COPY CONSTRUCTOR - -- tu jakby zadzialal -- maly obiekt na rejestr
/// 2# dodatkowy krok na poczatku z syncjronizacja --> punktu zczytajmy z __device__ memory 

///
/// Free Point
///

__device__ void evaluateForceIntensity_FreePoint(int tID, graph::Geometric* geometric, EvaluationContext* ec, graph::Tensor& mt) 
{
        graph::Point const&  a  = ec->getPoint(geometric->a);
        graph::Point const&  b  = ec->getPoint(geometric->b);
        graph::Point const&  p1 = ec->getPoint(geometric->p1);

        double d_a_p1 = abs(p1.minus(a).length()) * 0.1;
        double d_p1_b = abs(b.minus(p1).length()) * 0.1;

        // 8 = 4*2 (4 punkty kontrolne)

        //F12 - sily w sprezynach
        graph::Vector f12 = p1.minus(a).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(p1.minus(a).length() - d_a_p1);
        //F23
        graph::Vector f23 = b.minus(p1).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(b.minus(p1).length() - d_p1_b);

        //F1 - sily na poszczegolne punkty
        mt.setVector(tID + 0, 0, f12);
        //F2
        mt.setVector(tID + 2, 0, f23.minus(f12));
        //F3
        mt.setVector(tID + 4, 0, f23.product(-1));
}

///
/// Line 
///
   
__device__ void evaluateForceIntensity_Line(int tID, graph::Geometric* geometric, EvaluationContext* ec, graph::Tensor& mt) 
{
        graph::Point const&  a  = ec->getPoint(geometric->a);
        graph::Point const&  b  = ec->getPoint(geometric->b);
        graph::Point const&  p1 = ec->getPoint(geometric->p1);
        graph::Point const&  p2 = ec->getPoint(geometric->p2);    

        double d_a_p1   = abs(p1.minus(a).length());
        double d_p1_p2  = abs(p2.minus(p1).length());
        double d_p2_b   = abs(b.minus(p2).length());

        // 8 = 4*2 (4 punkty kontrolne)
        graph::Vector f12 = p1.minus(a).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(p1.minus(a).length() - d_a_p1);          //F12 - sily w sprezynach
        graph::Vector f23 = p2.minus(p1).unit().product(CONSTS_SPRING_STIFFNESS_HIGH).product(p2.minus(p1).length() - d_p1_p2);      //F23
        graph::Vector f34 = b.minus(p2).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(b.minus(p2).length() - d_p2_b);          //F34

        //F1 - silu na poszczegolne punkty
        mt.setVector(tID + 0, 0, f12);
        //F2
        mt.setVector(tID + 2, 0, f23.minus(f12));
        //F3
        mt.setVector(tID + 4, 0, f34.minus(f23));
        //F4
        mt.setVector(tID + 6, 0, f34.product(-1.0));
}

///
/// Circle
///

__device__ void evaluateForceIntensity_Circle(int tID, graph::Geometric* geometric, EvaluationContext* ec, graph::Tensor& mt) 
{
        graph::Point const&  a  = ec->getPoint(geometric->a);
        graph::Point const&  b  = ec->getPoint(geometric->b);
        graph::Point const&  p1 = ec->getPoint(geometric->p1);
        graph::Point const&  p2 = ec->getPoint(geometric->p2);    

        double d_a_p1   = abs(p1.minus(a).length());
        double d_p1_p2  = abs(p2.minus(p1).length());
        double d_p2_b   = abs(b.minus(p2).length());

        // 8 = 4*2 (4 punkty kontrolne)
        graph::Vector f12 = p1.minus(a).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(p1.minus(a).length() - d_a_p1);          //F12 - sily w sprezynach
        graph::Vector f23 = p2.minus(p1).unit().product(CONSTS_SPRING_STIFFNESS_HIGH * CIRCLE_SPRING_ALFA).product(p2.minus(p1).length() - d_p1_p2);      //F23
        graph::Vector f34 = b.minus(p2).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(b.minus(p2).length() - d_p2_b);          //F34

        //F1 - silu na poszczegolne punkty
        mt.setVector(tID + 0, 0, f12);
        //F2
        mt.setVector(tID + 2, 0, f23.minus(f12));
        //F3
        mt.setVector(tID + 4, 0, f34.minus(f23));
        //F4
        mt.setVector(tID + 6, 0, f34.product(-1.0));
}

///
/// Arc
///

__device__ void evaluateForceIntensity_Arc(int tID, graph::Geometric* geometric, EvaluationContext* ec, graph::Tensor& mt)
{
        graph::Point const&  a  = ec->getPoint(geometric->a);
        graph::Point const&  b  = ec->getPoint(geometric->b);
        graph::Point const&  c  = ec->getPoint(geometric->c);
        graph::Point const&  d  = ec->getPoint(geometric->d);
        graph::Point const&  p1 = ec->getPoint(geometric->p1);
        graph::Point const&  p2 = ec->getPoint(geometric->p2);    
        graph::Point const&  p3 = ec->getPoint(geometric->p3);    

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

        mt.setVector(row + 0, 0, fap1);
        mt.setVector(row + 2, 0, fbp1);
        mt.setVector(row + 4, 0, fp2c.product(-1));
        mt.setVector(row + 6, 0, fp3d.product(-1));
        mt.setVector(row + 8, 0, fp1p2.plus(fp1p3).minus(fap1).minus(fbp1));
        mt.setVector(row + 10, 0, fp2c.minus(fp1p2));
        mt.setVector(row + 12, 0, fp3d.minus(fp1p3));
}


///
/// ================================ STIFFNESS MATRIX ==================== ///
///

///
/// Free Point
///

__device__ void setStiffnessMatrix_FreePoint(int tID, graph::Tensor& mt) 
{
        /**
         * k= I*k
         * [ -ks    ks     0;
         *    ks  -2ks   ks ;
         *     0    ks   -ks];

         */
        // K -mala sztywnosci
        graph::SmallTensor Ks = graph::SmallTensor.diagonal(CONSTS_SPRING_STIFFNESS_LOW, CONSTS_SPRING_STIFFNESS_LOW);
        graph::SmallTensor Km = Ks.multiplyC(-1);

        mt.plusSubTensor(tID + 0, tID + 0, Km);
        mt.plusSubTensor(tID + 0, tID + 2, Ks);

        mt.plusSubTensor(tID + 2, tID + 0, Ks);
        mt.plusSubTensor(tID + 2, tID + 2, Km.multiplyC(2.0));
        mt.plusSubTensor(tID + 2, tID + 4, Ks);

        mt.plusSubTensor(tID + 4, tID + 2, Ks);
        mt.plusSubTensor(tID + 4, tID + 4, Km);
}

///
/// Line
///

__device__ void setStiffnessMatrix_Line(int tID, graph::Tensor& mt) 
{
        /**
         * k= I*k
         * [ -ks    ks      0  	  0;
         *    ks  -ks-kb   kb  	  0;
         *     0    kb   -ks-kb   ks;
         *     0  	 0     ks    -ks];
         */
        // K -mala sztywnosci
        graph::SmallTensor Ks = graph::SmallTensor.diagonal(CONSTS_SPRING_STIFFNESS_LOW, CONSTS_SPRING_STIFFNESS_LOW);
        // K - duza szytwnosci
        graph::SmallTensor Kb = graph::SmallTensor.diagonal(CONSTS_SPRING_STIFFNESS_HIGH, CONSTS_SPRING_STIFFNESS_HIGH);
        // -Ks-Kb
        graph::SmallTensor Ksb = Ks.multiplyC(-1).plus(Kb.multiplyC(-1));

        //wiersz pierwszy
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


///
/// Circle
///


__device__ void setStiffnessMatrix_Circle(int tID, graph::Tensor& mt) 
{
        /**
         * k= I*k
         * [ -ks    ks      0  	  0;
         *    ks  -ks-kb   kb  	  0;
         *     0    kb   -ks-kb   ks;
         *     0  	 0     ks    -ks];
         */
        // K -mala sztywnosci
        graph::SmallTensor Ks = graph::SmallTensor.diagonal(CONSTS_SPRING_STIFFNESS_LOW, CONSTS_SPRING_STIFFNESS_LOW);
        // K - duza szytwnosci
        graph::SmallTensor Kb = graph::SmallTensor.diagonal(CONSTS_SPRING_STIFFNESS_HIGH * CIRCLE_SPRING_ALFA, CONSTS_SPRING_STIFFNESS_HIGH * CIRCLE_SPRING_ALFA);
        // -Ks-Kb
        graph::SmallTensor Ksb = Ks.multiplyC(-1).plus(Kb.multiplyC(-1));

        //wiersz pierwszy
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
/// Arc
///

__device__ void setStiffnessMatrix_Arc(int tID, graph::Tensor& mt) 
{
        // K -mala sztywnosci
        graph::SmallTensor Kb = graph::SmallTensor.diagonal(CONSTS_SPRING_STIFFNESS_HIGH, CONSTS_SPRING_STIFFNESS_HIGH);
        graph::SmallTensor Ks = graph::SmallTensor.diagonal(CONSTS_SPRING_STIFFNESS_LOW, CONSTS_SPRING_STIFFNESS_LOW);

        graph::SmallTensor mKs = Ks.multiplyC(-1);
        graph::SmallTensor mKb = Kb.multiplyC(-1);
        graph::SmallTensor KsKbm = mKs.plus(mKb);

        mt.plusSubMatrix(tID + 0, tID + 0, mKs);
        mt.plusSubMatrix(tID + 0, tID + 8, Ks);//a

        mt.plusSubMatrix(tID + 2, tID + 2, mKs);
        mt.plusSubMatrix(tID + 2, tID + 8, Ks);//b

        mt.plusSubMatrix(tID + 4, tID + 4, mKs);
        mt.plusSubMatrix(tID + 4, tID + 10, Ks);//c

        mt.plusSubMatrix(tID + 6, tID + 6, mKs);
        mt.plusSubMatrix(tID + 6, tID + 12, Ks);//d

        mt.plusSubMatrix(tID + 8, tID + 0, Ks);
        mt.plusSubMatrix(tID + 8, tID + 2, Ks);
        mt.plusSubMatrix(tID + 8, tID + 8, KsKbm.multiplyC(2.0));
        mt.plusSubMatrix(tID + 8, tID + 10, Kb);
        mt.plusSubMatrix(tID + 8, tID + 12, Kb); //p1

        mt.plusSubMatrix(tID + 10, tID + 4, Ks);
        mt.plusSubMatrix(tID + 10, tID + 8, Kb);
        mt.plusSubMatrix(tID + 10, tID + 10, KsKbm); //p2

        mt.plusSubMatrix(tID + 12, tID + 6, Ks);
        mt.plusSubMatrix(tID + 12, tID + 8, Kb);
        mt.plusSubMatrix(tID + 12, tID + 12, KsKbm); //p3
}

