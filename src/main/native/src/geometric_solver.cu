#include "geometric_solver.h"

#include "stop_watch.h"
#include <cuda_runtime_api.h>
#include <memory>

#include <functional>
#include <numeric>

#include <stdexcept>
#include <stdio.h>

#include "model.cuh"
#include "model_config.h"

/// ------------ domain model przenosimy do pliku *.cu i laczymy assemblujemy *.cuh -> do *.cu

/// points register
static std::vector<graph::Point> points; /// poLocations id-> point_offset

/// geometricc register
static std::vector<graph::Geometric> geometrics; /// ==> Macierz A, accumulative offset for each primitive

/// constraints register
static std::vector<graph::Constraint> constraints; /// ===> Wiezy , accumulative offset for each constraint

/// parameters register
static std::vector<graph::Parameter> parameters; /// paramLocation id-> param_offset

/// Point  Offset in computation matrix [id] -> point offset   ~~ Gather Vectors
static std::unique_ptr<int[]> pointOffset;

/// Accymulated Geometric Object Size
static std::unique_ptr<int[]> accGeometricSize; /// 2 * point.size()

/// Accumulated Constraint Size
static std::unique_ptr<int[]> accConstraintSize;


/// Solver performance watchers
static graph::StopWatch solverWatch;
static graph::StopWatch accEvoWatch;
static graph::StopWatch accLUWatch;


int size;      /// wektor stanu
int coffSize;  /// wspolczynniki Lagrange
int dimension; /// dimension = size + coffSize


template <typename Obj, typename ObjIdFunction> std::unique_ptr<int[]> stateOffset(std::vector<Obj> objects, ObjIdFunction objectIdFunction) {
        std::unique_ptr<int[]> offsets(new int[objectIdFunction(objects.rbegin())]);
        auto iterator = objects.begin();
        int offset = 0;
        while (iterator != objects.end()) {
                auto objectId = objectIdFunction(iterator);
                offsets[objectId] = offset++;                
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


/// KERNEL#
///         --- allocate in `setup` function  , all references are valid GPU memory references
///
///         cudaMalloc, cudaMemcpy
///
struct Evaluation {

        int size;      /// wektor stanu
        int coffSize;  /// wspolczynniki Lagrange
        int dimension; /// N - dimension = size + coffSize

/// macierze ukladu rowañ
        double *A;
        double *x;
        double *dx;
        double *b;

/// mainly static data
        graph::Point *points;            
        graph::Geometric *geometrics;
        graph::Constraint *constraints;
        graph::Parameter *parameters;

        const int * pointOffset;      /// ---
        const int * accGeometricSize;  /// accumulative offset with geometric size evaluation function        
        const int * accConstraintSize; /// accumulative offset with constraint size evaluation function

        /// w drugiej fazie dopisac => dodatkowo potrzebny per block  __shared__  memory   ORR    L1   fast region memory

        __host__ __device__ graph::Point const &getPoint(int pointId) const;

        __host__ __device__ graph::Geometric *getGeomtricObject(int geometricId) const;
};



__device__ graph::Point const &Evaluation::getPoint(int pointId) const {
        int offset = pointOffset[pointId];
        return points[offset];
};

__device__ graph::Geometric *Evaluation::getGeomtricObject(int geometricId) const { return const_cast<graph::Geometric *>(&geometrics[geometricId]); };



/// KERNEL# GPU
__global__ void kernel_add(int *HA, int *HB, int *HC, int size);


/// [ KERNEL# GPU ]

__device__ void setStiffnessMatrix_FreePoint(int tid, graph::Tensor &mt);

__device__ void setStiffnessMatrix_Line(int tid, graph::Tensor &mt);

__device__ void setStiffnessMatrix_FixLine(int tid, graph::Tensor &mt);

__device__ void setStiffnessMatrix_Circle(int tid, graph::Tensor &mt);

__device__ void setStiffnessMatrix_Arc(int tid, graph::Tensor &mt);

__global__ void computeStiffnessMatrix(Evaluation *ec, int N);



__device__ void evaluateForceIntensity_FreePoint(int tID, graph::Geometric *geometric, Evaluation *ec, graph::Tensor &mt);

__device__ void evaluateForceIntensity_Line(int tID, graph::Geometric *geometric, Evaluation *ec, graph::Tensor &mt);

__device__ void evaluateForceIntensity_FixLine(int tID, graph::Geometric *geometric, Evaluation *ec, graph::Tensor &mt);

__device__ void evaluateForceIntensity_Circle(int tID, graph::Geometric *geometric, Evaluation *ec, graph::Tensor &mt);

__device__ void evaluateForceIntensity_Arc(int tID, graph::Geometric *geometric, Evaluation *ec, graph::Tensor &mt);

__global__ void evaluateForceIntensity(Evaluation *ec, int N);




namespace solver {

void resetComputationData(cudaError_t *error) {
        std::remove_if(points.begin(), points.end(), [](auto _) { return true; });
        std::remove_if(geometrics.begin(), geometrics.end(), [](auto _) { return true; });
        std::remove_if(constraints.begin(), constraints.end(), [](auto _) { return true; });
        std::remove_if(parameters.begin(), parameters.end(), [](auto _) { return true; });

        pointOffset = NULL;
        accConstraintSize = NULL;
        accGeometricSize = NULL;
}


/**
 *
 */
void resetComputationContext(cudaError_t *error) {}

/**
 *
 */
void initComputationContext(cudaError_t *error) 
{     
    if (points.size() == 0){
        throw new std::exception("empty solution space, add some geometric types");
    }
  
    
    /// mapping from point Id => point vector offset
    pointOffset = stateOffset(points, [](auto point) { return point->id; });
    /// accumalted position of geometric block
    accGeometricSize = accumalatedValue(geometrics, graph::geometricSetSize);
    /// accumulated position of constrain block
    accConstraintSize = accumalatedValue(constraints, graph::constraintSize);



    solverWatch.reset();
    accEvoWatch.reset();
    accLUWatch.reset();

    /// `A` tensor internal structure dimensions
    size = std::accumulate(geometrics.begin(), geometrics.end(), 0, [](auto acc, auto const &geometric) { 
        return acc + graph::geometricSetSize(geometric); 
    });
    coffSize = std::accumulate(constraints.begin(), constraints.end(), 0, [](auto acc, auto const &constraint) { 
        return acc + graph::constraintSize(constraint); 
    });
    
    dimension = size + coffSize;
    
    /**
     * po zarejestrowaniu calego modelu w odpowiadajacych rejestrach , zainicjalizowac pomocnicze macierze
     *
     * 
     *
     * przeliczenie pozycji absolutnej punktu na macierzy wyjsciowej
     */

}


/**
 *
 */
void registerPointType(int id, double px, double py) { points.emplace_back(id, px, py); }

/**
 *
 */
void registerGeometricType(int id, int geometricTypeId, int p1, int p2, int p3, int a, int b, int c, int d) {
        geometrics.emplace_back(id, geometricTypeId, p1, p2, p3, a, b, c, d);
}

/**
 *
 */
void registerParameterType(int id, double value) { parameters.emplace_back(id, value);  }

/**
 *
 */
void registerConstraintType(int id, int jconstraintTypeId, int k, int l, int m, int n, int paramId, double vecX, double vecY) 
{
        constraints.emplace_back(id, jconstraintTypeId, k, l, m, n, paramId, vecX, vecY);
}

/**
 *
 */
double getPointPXCoordinate(int id) 
{
        int offset = pointOffset[id];
        double px = points[offset].x;
        return px;
}

/**
 *
 */
double getPointPYCoordinate(int id)
{
        int offset = pointOffset[id];
        double py = points[offset].y;
        return py;
}


        
void getPointCoordinateVector(double *state_arr) 
{
   

}





///
/// Setup all matricies for computation and prepare kernel stream  intertwined with cuSolver
///
///
void solveSystemOnGPU(solver::SolverStat *stat, cudaError_t *error) 
{
        int N = dimension;
        
/// Uklad rownan liniowych  [ A * x = b ] powsta³y z linerazycji ukladu dynamicznego - tensory na urzadzeniu.
        double *dev_A;
        double *dev_b;
        double *dev_x;
        double *dev_dx;


        /// w drugiej fazie dopisac => dodatkowo potrzebny per block  __shared__  memory   ORR    L1   fast region memory
        /// 
/// Evaluation data for  device  - CONST DATE for in process execution

        graph::Point *d_points;
        graph::Geometric *d_geometrics;         
        graph::Constraint *d_constraints;
        graph::Parameter *d_parameters;

        int *d_pointOffset;       /// ---
        int *d_accGeometricSize;  /// accumulative offset with geometric size evaluation function
        int *d_accConstraintSize; /// accumulative offset with constraint size evaluation function

///
/// for [GPU] Evaluation  - dane obliczeniowego modelu !
///
        std::unique_ptr<Evaluation> ev(new Evaluation());
        ev->size = size;
        ev->coffSize = coffSize;
        ev->dimension = dimension = N;

        checkCudaStatus(cudaMalloc((void**)&d_points, points.size() * sizeof(graph::Point)));
        checkCudaStatus(cudaMalloc((void**)&d_geometrics, geometrics.size() * sizeof(graph::Geometric)));
        checkCudaStatus(cudaMalloc((void**)&d_constraints, constraints.size() * sizeof(graph::Constraint)));
        checkCudaStatus(cudaMalloc((void**)&d_parameters, parameters.size() * sizeof(graph::Parameter)));

        checkCudaStatus(cudaMemcpy(d_points, points.data(), points.size() * sizeof(graph::Point), cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(d_geometrics, geometrics.data(), geometrics.size() * sizeof(graph::Geometric), cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(d_constraints, constraints.data(), constraints.size() * sizeof(graph::Constraint), cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(d_parameters, parameters.data(), parameters.size() * sizeof(graph::Parameter), cudaMemcpyHostToDevice));

        ev->points = d_points;
        ev->geometrics= d_geometrics;
        ev->constraints= d_constraints;
        ev->parameters = d_parameters;
        
        checkCudaStatus(cudaMalloc((void**)&d_pointOffset, points.rbegin()->id * sizeof(int)));
        checkCudaStatus(cudaMalloc((void**)&d_accGeometricSize, geometrics.size() * sizeof(int)));
        checkCudaStatus(cudaMalloc((void**)&d_accConstraintSize, constraints.size() * sizeof(int)));

        checkCudaStatus(cudaMemcpy(d_pointOffset, pointOffset.get(), points.rbegin()->id * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(d_accGeometricSize, accGeometricSize.get(), geometrics.size() * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(d_accConstraintSize, accConstraintSize.get(), constraints.size() * sizeof(int), cudaMemcpyHostToDevice));
            
        ev->pointOffset = d_pointOffset;
        ev->accGeometricSize= d_accGeometricSize;
        ev->accConstraintSize = d_accConstraintSize;
                     
///
///  [ GPU ] tensors `A` `x` `dx` `b`  ------------------------
/// 
        
        checkCudaStatus(cudaMalloc((void **)&dev_A, N * N * sizeof(double)));
        checkCudaStatus(cudaMalloc((void **)&dev_b, N * sizeof(double)));
        checkCudaStatus(cudaMalloc((void **)&dev_x, N * sizeof(double)));
        checkCudaStatus(cudaMalloc((void **)&dev_dx, N * sizeof(double)));

        checkCudaStatus(cudaMemset((void **)&dev_x, 0.0, N));
        checkCudaStatus(cudaMemset((void **)&dev_dx, 0.0, N));

        ev->A   = dev_A;
        ev->x   = dev_x;
        ev->dx  = dev_dx;
        ev->b   = dev_b;
        
///
        /// [ GPU ] Evalution
        ///
        ///
        Evaluation *dev_ev;

        checkCudaStatus(cudaMalloc((void **)&dev_ev, sizeof(Evaluation)));
        checkCudaStatus(cudaMemcpy(dev_ev, ev.get(), sizeof(Evaluation), cudaMemcpyHostToDevice));               


/// referencec         
        //graph::Tensor A = graph::Tensor::fromDeviceMem(dev_A, N, N); /// Macierz g³owna ukladu rownan liniowych
        
        //graph::Tensor Fq; /// [size x size]     Macierz sztywnosci ukladu obiektow zawieszonych na sprezynach.
        //graph::Tensor Wq; /// [coffSize x size]  d(FI)/dq - Jacobian Wiezow

        /// HESSIAN
        //graph::Tensor Hs;

        // Wektor prawych stron [Fr; Fi]'
        //graph::Tensor b = graph::Tensor::fromDeviceMem(dev_b, N, 1);

        // skladowe to Fr - oddzialywania pomiedzy obiektami, sily w sprezynach
        //graph::Tensor Fr = graph::Tensor::fromDeviceMem(dev_b, size , 1);

        // skladowa to Fiq - wartosci poszczegolnych wiezow
        //graph::Tensor Fi = graph::Tensor::fromDeviceMem(dev_b + size, coffSize, 1);



        double norm1;            /// wartosci bledow na wiezach
        double prevNorm;         /// norma z wczesniejszej iteracji,
        double errorFluctuation; /// fluktuacja bledu

        stat->startTime = graph::TimeNanosecondsNow();



        printf("@#=================== Solver Initialized ===================#@ \n");
        printf("");

        int blockSize = geometrics.size();
        computeStiffnessMatrix<<<1, blockSize>>>(dev_ev, blockSize);



        while(true)
        {
            /// compute next approximation
        
        }


        *error = cudaSuccess;
}

} // namespace solver



__global__ void kernel_add(int *HA, int *HB, int *HC, int size)
{
        int i = threadIdx.x;
        if (i < size) 
        {
                HC[i] = HA[i] + HB[i];
        }
}



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


/// DEVICE# context


/// @brief Setup __device__ vector of geometric points in this moment.
///
/// @param ec evaluation context
/// @param N size of geometric object data vector
/// @param _point[] __shared__ reference into model point state
/// @tparam TZ tensor dimension without constraints
/// @return void
///
__device__ void SetupPointAccessor(Evaluation *ec, int GN, /* __shared__*/ double *_points) {
        
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
                        printf("[error] type of geometric not recognized \n");
                        break;
                }
        }
}


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
__global__ void computeStiffnessMatrix(Evaluation *ec, int N) {
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
                        printf("[error] type of geometric not recognized \n");
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
/// Evaluate Force Intensity ==============================================================
///

__global__ void evaluateForceIntensity(Evaluation *ec, int N) {
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


///
/// Free Point ============================================================================
///

__device__ void evaluateForceIntensity_FreePoint(int tID, graph::Geometric *geometric, Evaluation *ec, graph::Tensor &mt) 
{
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

__device__ void evaluateForceIntensity_Line(int tID, graph::Geometric *geometric, Evaluation *ec, graph::Tensor &mt) 
{
        graph::Point const &a = ec->getPoint(geometric->a);
        graph::Point const &b = ec->getPoint(geometric->b);
        graph::Point const &p1 = ec->getPoint(geometric->p1);
        graph::Point const &p2 = ec->getPoint(geometric->p2);

        double d_a_p1 = abs(p1.minus(a).length());
        double d_p1_p2 = abs(p2.minus(p1).length());
        double d_p2_b = abs(b.minus(p2).length());

        // 8 = 4*2 (4 punkty kontrolne)
        graph::Vector f12 = p1.minus(a).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(p1.minus(a).length() - d_a_p1);     // F12 - sily w sprezynach
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

__device__ void evaluateForceIntensity_FixLine(int tID, graph::Geometric *geometric, Evaluation *ec, graph::Tensor &mt) {}

///
/// Circle  ===============================================================================
///

__device__ void evaluateForceIntensity_Circle(int tID, graph::Geometric *geometric, Evaluation *ec, graph::Tensor &mt) 
{
        graph::Point const &a = ec->getPoint(geometric->a);
        graph::Point const &b = ec->getPoint(geometric->b);
        graph::Point const &p1 = ec->getPoint(geometric->p1);
        graph::Point const &p2 = ec->getPoint(geometric->p2);

        double d_a_p1 = abs(p1.minus(a).length());
        double d_p1_p2 = abs(p2.minus(p1).length());
        double d_p2_b = abs(b.minus(p2).length());

        // 8 = 4*2 (4 punkty kontrolne)
        graph::Vector f12 = p1.minus(a).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(p1.minus(a).length() - d_a_p1); // F12 - sily w sprezynach
        graph::Vector f23 = p2.minus(p1).unit().product(CONSTS_SPRING_STIFFNESS_HIGH * CIRCLE_SPRING_ALFA).product(p2.minus(p1).length() - d_p1_p2); // F23
        graph::Vector f34 = b.minus(p2).unit().product(CONSTS_SPRING_STIFFNESS_LOW).product(b.minus(p2).length() - d_p2_b);                          // F34

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

__device__ void evaluateForceIntensity_Arc(int tID, graph::Geometric *geometric, Evaluation *ec, graph::Tensor &mt) 
{
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



/// <summary>
///
/// </summary>
/// <param name="status"></param>
/// <param name="__line_"></param>
void checkCudaStatus_impl(cudaError_t status, size_t __line_) 
{
        if (status != cudaSuccess)
        {
                printf("%d: cuda API failed with status %d\n", static_cast<int>(__line_), static_cast<int>(status));
                throw std::logic_error("cuda API error");
        }
}