#include "geometric_solver.h"

#include "cuda_runtime.h"
#include "stop_watch.h"

#include <memory>

#include <functional>
#include <numeric>

#include <stdexcept>
#include <stdio.h>

#include "model.cuh"
#include "solver_kernel.cuh"

#include "linear_system.h"

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

/// Parameter Offset mapper from [id] -> parameter offset in reference vector
static std::unique_ptr<int[]> parameterOffset;

/// Accymulated Geometric Object Size
static std::unique_ptr<int[]> accGeometricSize; /// 2 * point.size()

/// Accumulated Constraint Size
static std::unique_ptr<int[]> accConstraintSize;

/// === Solver Performance Watchers

/// observation of total calculation time
static graph::StopWatch solverWatch;

/// observation all neccessar evalution of tensor A,b and tensor views.
static graph::StopWatch accEvoWatch;

/// observation of accumalated cuSolver method
static graph::StopWatch accLUWatch;

int size;      /// wektor stanu
int coffSize;  /// wspolczynniki Lagrange
int dimension; /// dimension = size + coffSize

namespace utility {

template <typename Ty> 
void memcpyToDevice(Ty **dest_device, const Ty *const &vector, size_t size) {
    /// allocate destination vector
    checkCudaStatus(cudaMalloc((void **)dest_device, size * sizeof(Ty)));

    /// transfer into new allocation host_vector
    checkCudaStatus(cudaMemcpy(*dest_device, vector, size * sizeof(Ty), cudaMemcpyHostToDevice));
}

template <typename Ty> 
void memcpyToDevice(Ty **dest_device, std::vector<Ty> const &vector) {
    /// memcpy to device
    memcpyToDevice(dest_device, vector.data(), vector.size());
}

template <typename Ty> void mallocToDevice(Ty **dev_ptr, size_t size) {
    /// safe malloc
    checkCudaStatus(cudaMalloc((void **)dev_ptr, size));
}



template <typename Ty> 
void memcpyFromDevice(std::vector<Ty> &vector, Ty *src_device) {
    /// transfer into new allocation host_vector
    checkCudaStatus(cudaMemcpy(vector.data(), src_device, vector.size() * sizeof(Ty), cudaMemcpyDeviceToHost));
}




template <typename Ty> void freeMem(Ty *dev_ptr) {
    /// safe free mem
    checkCudaStatus(cudaFree(dev_ptr));
}

void memset(void *dev_ptr, int value, size_t size) {
    ///
    checkCudaStatus(cudaMemset(dev_ptr, value, size));
}


template <typename Obj, typename ObjIdFunction>
std::unique_ptr<int[]> stateOffset(std::vector<Obj> objects, ObjIdFunction objectIdFunction) {

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

template <typename Obj, typename ValueFunction>
std::unique_ptr<int[]> accumalatedValue(std::vector<Obj> vector, ValueFunction valueFunction) {
    int accValue = 0;
    std::unique_ptr<int[]> accumulated(new int[vector.size()]);
    for (int offset = 0; offset < vector.size(); offset++) {
        accumulated[offset] = accValue;
        accValue = accValue + valueFunction(vector[offset]);
    }
    return accumulated;
}

} // namespace utility

/// KERNEL# GPU
__global__ void kernel_add(int *HA, int *HB, int *HC, int size) {
    int i = threadIdx.x;
    if (i < size) {
        HC[i] = HA[i] + HB[i];
    }
}

namespace solver {

void resetComputationData(cudaError_t *error) {
    std::remove_if(points.begin(), points.end(), [](auto _) { return true; });
    std::remove_if(geometrics.begin(), geometrics.end(), [](auto _) { return true; });
    std::remove_if(constraints.begin(), constraints.end(), [](auto _) { return true; });
    std::remove_if(parameters.begin(), parameters.end(), [](auto _) { return true; });

    pointOffset = NULL;
    parameterOffset = NULL;
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
void initComputationContext(cudaError_t *error) {
    if (points.size() == 0) {
        throw new std::exception("empty solution space, add some geometric types");
    }

    /// mapping from point Id => point vector offset
    pointOffset = utility::stateOffset(points, [](auto point) { return point->id; });
    /// this is mapping from Id => parameter vector offset
    parameterOffset = utility::stateOffset(parameters, [](auto parameter) { return parameter->id; });
    /// accumalted position of geometric block
    accGeometricSize = utility::accumalatedValue(geometrics, graph::geometricSetSize);
    /// accumulated position of constrain block
    accConstraintSize = utility::accumalatedValue(constraints, graph::constraintSize);

    solverWatch.reset();
    accEvoWatch.reset();
    accLUWatch.reset();

    /// `A` tensor internal structure dimensions
    size = std::accumulate(geometrics.begin(), geometrics.end(), 0,
                           [](auto acc, auto const &geometric) { return acc + graph::geometricSetSize(geometric); });
    coffSize = std::accumulate(constraints.begin(), constraints.end(), 0,
                               [](auto acc, auto const &constraint) { return acc + graph::constraintSize(constraint); });

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
void registerParameterType(int id, double value) { parameters.emplace_back(id, value); }

/**
 *
 */
void registerConstraintType(int id, int jconstraintTypeId, int k, int l, int m, int n, int paramId, 
    double vecX, double vecY) 
{
    constraints.emplace_back(id, jconstraintTypeId, k, l, m, n, paramId, vecX, vecY);
}

/**
 *
 */
double getPointPXCoordinate(int id) {
    int offset = pointOffset[id];
    double px = points[offset].x;
    return px;
}

/**
 *
 */
double getPointPYCoordinate(int id) {
    int offset = pointOffset[id];
    double py = points[offset].y;
    return py;
}

void getPointCoordinateVector(double *state_arr) {}

#define MAX_SOLVER_ITERATIONS 20

#define CONVERGENCE_LIMIT 10e-5

///
/// Setup all matricies for computation and prepare kernel stream  intertwined with cuSolver
///
///
void solveSystemOnGPU(solver::SolverStat *stat, cudaError_t *error) {
    int N = dimension;

    /// Uklad rownan liniowych  [ A * x = b ] powsta³y z linerazycji ukladu dynamicznego - tensory na urzadzeniu.
    double *dev_A = nullptr;
    double *dev_b = nullptr;
    double *dev_SV = nullptr; /// STATE VECTOR
    double *dev_dx = nullptr; /// [ A ] * [ dx ] = [ b ]

    /// w drugiej fazie dopisac => dodatkowo potrzebny per block  __shared__  memory   ORR    L1   fast region memory
    ///
    /// Evaluation data for  device  - CONST DATE for in process execution

    graph::Point *d_points = nullptr;
    graph::Geometric *d_geometrics = nullptr;
    graph::Constraint *d_constraints = nullptr;
    graph::Parameter *d_parameters = nullptr;

    int *d_pointOffset;       /// ---
    int *d_parameterOffset;   ///
    int *d_accGeometricSize;  /// accumulative offset with geometric size evaluation function
    int *d_accConstraintSize; /// accumulative offset with constraint size evaluation function

    if (points.size() == 0) {
        printf("[solver] - empty evaluation space model\n");
        return;
    }

    if (constraints.size()) {
        printf("[solver] - no constraint configuration applied onto model\n");
        return;
    }

/// cuSolver component configuration
    int major = 0;    
    int minor = 0;
    int patch = 0;
    
    checkCuSolverStatus(cusolverGetProperty(MAJOR_VERSION, &major));
    checkCuSolverStatus(cusolverGetProperty(MINOR_VERSION, &minor));
    checkCuSolverStatus(cusolverGetProperty(PATCH_LEVEL, &patch));
    printf("CUSOLVER Version (Major,Minor,PatchLevel): %d.%d.%d\n", major, minor, patch);


    ///
    /// for [GPU] Evaluation  - dane obliczeniowego modelu !
    ///
    std::unique_ptr<Computation> ev(new Computation());
    ev->size = size;
    ev->coffSize = coffSize;
    ev->dimension = dimension = N;

    utility::memcpyToDevice(&d_points, points);
    utility::memcpyToDevice(&d_geometrics, geometrics);
    utility::memcpyToDevice(&d_constraints, constraints);
    utility::memcpyToDevice(&d_parameters, parameters);

    utility::memcpyToDevice(&d_pointOffset, pointOffset.get(), points.rbegin()->id);
    utility::memcpyToDevice(&d_parameterOffset, parameterOffset.get(), parameters.rbegin()->id);
    utility::memcpyToDevice(&d_accGeometricSize, accGeometricSize.get(), geometrics.size());
    utility::memcpyToDevice(&d_accConstraintSize, accConstraintSize.get(), constraints.size());

    ev->points = d_points;
    ev->geometrics = d_geometrics;
    ev->constraints = d_constraints;
    ev->parameters = d_parameters;

    ev->pointOffset = d_pointOffset;
    ev->parameterOffset = d_parameterOffset;
    ev->accGeometricSize = d_accGeometricSize;
    ev->accConstraintSize = d_accConstraintSize;

    ///
    ///  [ GPU ] tensors `A` `x` `dx` `b`  ------------------------
    ///

    utility::mallocToDevice(&dev_A, N * N * sizeof(double));
    utility::mallocToDevice(&dev_b, N * sizeof(double));
    utility::mallocToDevice(&dev_SV, size * sizeof(double));
    utility::mallocToDevice(&dev_dx, N * sizeof(double));

    ev->A = dev_A;
    ev->b = dev_b;
    ev->SV = dev_SV;
    ev->dx = dev_dx;

    ///
    /// [ GPU ] computation context mapped onto devive object
    ///
    ///
    Computation *dev_ev;

    utility::memcpyToDevice(&dev_ev, ev.get(), 1);

    /// referencec
    // graph::Tensor A = graph::Tensor::fromDeviceMem(dev_A, N, N); /// Macierz g³owna ukladu rownan liniowych

    // graph::Tensor Fq; /// [size x size]     Macierz sztywnosci ukladu obiektow zawieszonych na sprezynach.
    // graph::Tensor Wq; /// [coffSize x size]  d(FI)/dq - Jacobian Wiezow

    /// HESSIAN
    // graph::Tensor Hs;

    // Wektor prawych stron [Fr; Fi]'
    // graph::Tensor b = graph::Tensor::fromDeviceMem(dev_b, N, 1);

    // skladowe to Fr - oddzialywania pomiedzy obiektami, sily w sprezynach
    // graph::Tensor Fr = graph::Tensor::fromDeviceMem(dev_b, size , 1);

    // skladowa to Fiq - wartosci poszczegolnych wiezow
    // graph::Tensor Fi = graph::Tensor::fromDeviceMem(dev_b + size, coffSize, 1);

    double norm1;            /// wartosci bledow na wiezach
    double prevNorm;         /// norma z wczesniejszej iteracji,
    double errorFluctuation; /// fluktuacja bledu

    stat->startTime = graph::TimeNanosecondsNow();

    printf("@#=================== Solver Initialized ===================#@ \n");
    printf("");

    /// Inicjalizacja Macierzy A, b, i pochodne

    accEvoWatch.setStartTick();

    /// KERNEL_0 - Cooficients Stiffnes Matrix
    int blockSize = geometrics.size();
    computeStiffnessMatrix<<<1, 1024>>>(dev_ev, blockSize);

    /// asynchronous kernel invocation
    checkCudaStatus(cudaDeviceSynchronize());

    CopyIntoStateVector<<<1, 1024>>>(dev_ev);

    // ??SetupLagrangeMultipliers(); // SV

    accEvoWatch.setStopTick();

    cudaError_t lastError = cudaPeekAtLastError();
    if (lastError != cudaSuccess) {
        printf(" [ kernel ] computation failed - stiffness matrix \n");
        throw std::logic_error("kernel - stiffness matrix");
    }

    norm1 = 0.0;
    prevNorm = 0.0;
    errorFluctuation = 0.0;

    int itr = 0;

    linear_system_method_0_reset();


    while (itr < MAX_SOLVER_ITERATIONS) {

        accEvoWatch.setStartTick();

        /// --- KERNEL_PRE

        /// zerujemy macierz A

        /// # KERNEL_PRE

        utility::memset(dev_ev->A, 0, N * N * sizeof(double)); // --- ze wzgledu na addytywnosc

        /// Tworzymy Macierz vector b vector `b

        /// # KERNEL_PRE
        GeometricObjectEvaluateForceVector(); // Fr /// Sily  - F(q)
        ConstraintEvaluateConstraintVector(); // Fi / Wiezy  - Fi(q)

        // b.setSubMatrix(0,0, (Fr));
        // b.setSubMatrix(size,0, (Fi));

        b; /// --- schowac Do EvaluteForce - Constraint  .mulitply(-1);

        /// macierz `A

        /// # KERNEL_PRE (__shared__ JACOBIAN)

        ConstraintGetFullJacobian(); // --- (Wq); /// Jq = d(Fi)/dq --- write without intermediary matrix   `A = `A set value


        /// # KERNEL_PRE

        ConstraintGetFullHessian(); // --- (Hs, SV, size); // --- write without intermediate matrix '`A = `A +value

        A.setSubTensor(0, 0, Fq);  /// procedure SET
        A.plusSubTensor(0, 0, Hs); /// procedure ADD

        A.setSubTensor(size, 0, Wq);
        A.setSubTensor(0, size, Wq.transpose());

        accEvoWatch.setStopTick();


        /// DENSE - CuSolver
        /// LU Solver
        /// 
/// ======== LINER SYSTEM equation CuSolver    === START
        
        accLUWatch.setStartTick();

        
        linear_system_method_0(dev_ev->A, dev_ev->b, dev_ev->dimension);


        accLUWatch.setStopTick();

/// ======== LINER SYSTEM equation CuSolver    === STOP
      
        /// uaktualniamy punkty [ SV ] = [ SV ] + [ delta ]
        
        /// --- KERNEL_POST - cublas saxpy !!!
        
        StateVectorAddDifference<<<1, 1024>>>(dev_ev->SV, dev_ev->b, dev_ev->dimension);


        /// --- KERNEL_POST( __synchronize__ REDUCTION )

///  get `constraint norm` 
/// 
        norm1 = ConstraintGetFullNorm();        

        printf(" [ step :: %02d]  duration [ns] = %,12d  norm = %e \n", itr, (accLUWatch.stopTick - accEvoWatch.startTick), norm1);

        /// Gdy po 5-6 przejsciach iteracji, normy wiezow kieruja sie w strone minimum energii, to repozycjonowac prowadzace punkty

        if (norm1 < CONVERGENCE_LIMIT) {
            stat->error = norm1;
            printf("fast convergence - norm [ %e ] \n", norm1);
            printf("constraint error = %e \n", norm1);
            printf("");
            break;
        }

        /// liczymy zmiane bledu
        errorFluctuation = norm1 - prevNorm;
        prevNorm = norm1;
        stat->error = norm1;

        if (itr > 1 && errorFluctuation / prevNorm > 0.70) {

            linear_system_method_0_reset();

            CopyFromStateVector<<<1, 1024>>>(dev_ev);
            

            utility::memcpyFromDevice(points, d_points);
            

            printf("CHANGES - STOP ITERATION *******");
            printf(" errorFluctuation          : %d \n", errorFluctuation);
            printf(" relative error            : %f \n", (errorFluctuation / norm1));
            solverWatch.setStopTick();
            stat->constraintDelta = ConstraintGetFullNorm();
            stat->convergence = false;
            stat->stopTime = graph::TimeNanosecondsNow();
            stat->iterations = itr;
            stat->accSolverTime = accLUWatch.accTime;
            stat->accEvaluationTime = accEvoWatch.accTime;
            stat->timeDelta = solverWatch.stopTick - solverWatch.startTick;
            return;
        }
        itr++;
    }

    solverWatch.setStopTick();
    long solutionDelta = solverWatch.delta();

    printf("# solution delta : %d \n", solutionDelta); // print execution time
    printf("\n");                                      // print execution time

    CopyFromStateVector<<<1, 1024>>>(dev_ev);

    utility::memcpyFromDevice(points, d_points);

    stat->constraintDelta = ConstraintGetFullNorm();
    stat->convergence = norm1 < CONVERGENCE_LIMIT;
    stat->stopTime = graph::TimeNanosecondsNow();
    stat->iterations = itr;
    stat->accSolverTime = accLUWatch.accTime;
    stat->accEvaluationTime = accEvoWatch.accTime;
    stat->timeDelta = solverWatch.stopTick - solverWatch.startTick;

    /// w drugiej fazie dopisac => dodatkowo potrzebny per block  __shared__  memory   ORR    L1   fast region memory
    ///
    /// Evaluation data for  device  - CONST DATE for in process execution


/// free allocated memory in solution
Error:
    utility::freeMem(dev_ev);

    utility::freeMem(dev_A);
    utility::freeMem(dev_b);
    utility::freeMem(dev_SV);
    utility::freeMem(dev_dx);

    utility::freeMem(d_points);
    utility::freeMem(d_geometrics);
    utility::freeMem(d_constraints);
    utility::freeMem(d_parameters);

    utility::freeMem(d_pointOffset);       /// point offset table
    utility::freeMem(d_accGeometricSize);  /// accumulative offset with geometric size evaluation function
    utility::freeMem(d_accConstraintSize); /// accumulative offset with constraint size evaluation function

    *error = cudaSuccess;
}

} // namespace solver

/// CPU#
long AllLagrangeCoffSize() { return 0; }

/// KERNEL#
double ConstraintGetFullNorm() { 


    // cublasStatus_t  cublasDnrm2(cublasHandle_t handle, int n, const double          *x, int incx, double *result)
}

/// CPU#
void PointLocationSetup() {}


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

/// <summary>
///
/// </summary>
/// <param name="status"></param>
/// <param name="__line_"></param>
void checkCudaStatus_impl(cudaError_t status, size_t __line_) {
    if (status != cudaSuccess) {
        printf("%d: cuda API failed with status %d\n", static_cast<int>(__line_), static_cast<int>(status));
        throw std::logic_error("cuda API error");
    }
}