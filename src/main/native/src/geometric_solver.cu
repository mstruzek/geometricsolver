#include "geometric_solver.h"

#include "cuda_runtime.h"
#include "stop_watch.h"

#include <memory>
#include <condition_variable>
#include <atomic>

#include <functional>
#include <numeric>

#include <stdexcept>
#include <stdio.h>

#include "model.cuh"
#include "solver_kernel.cuh"

#include "linear_system.h"

#include "model_config.h"


/* clang_format : LLVM,
                  GNU,
                  Google,
                  Chromium,
                  Microsoft,
                  Mozilla,
                  WebKit.
                  */


/// MAX SOLVER ITERATIONS
#define CMAX 20


#define CONVERGENCE_LIMIT 10e-5


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

/// observation of submited tasks
static graph::StopWatch solverWatch;


/// observation of computation time - single computation run
cudaEvent_t computeStart[CMAX] = {};
cudaEvent_t computeStop[CMAX] = {};


/// observation of matrices  preperations 
cudaEvent_t prepStart[CMAX] = {};
cudaEvent_t prepStop[CMAX] = {};


/// observation of accumalated cuSolver method
cudaEvent_t solverStart[CMAX] = {};
cudaEvent_t solverStop[CMAX] = {};


int size;      /// wektor stanu
int coffSize;  /// wspolczynniki Lagrange
int dimension; /// dimension = size + coffSize


/// ===========================================================
/// Async Computation - Implicit ref in utility:: namespace
///
static cudaStream_t stream = NULL;

///
/// ===========================================================


namespace utility {

template <typename Ty> 
void memcpyToDevice(Ty **dest_device, const Ty *const &vector, size_t size) {
    /// allocate destination vector
    checkCudaStatus(cudaMallocAsync((void **)dest_device, size * sizeof(Ty), stream));

    /// transfer into new allocation host_vector
    checkCudaStatus(cudaMemcpyAsync(*dest_device, vector, size * sizeof(Ty), cudaMemcpyHostToDevice, stream));
}

template <typename Ty> 
void memcpyToDevice(Ty **dest_device, std::vector<Ty> const &vector) {
    /// memcpy to device
    memcpyToDevice(dest_device, vector.data(), vector.size());
}

template <typename Ty> void mallocToDevice(Ty **dev_ptr, size_t size) {
    /// safe malloc
    checkCudaStatus(cudaMallocAsync((void **)dev_ptr, size, stream));
}



template <typename Ty> 
void memcpyFromDevice(std::vector<Ty> &vector, Ty *src_device) {
    /// transfer into new allocation host_vector
    checkCudaStatus(cudaMemcpyAsync(vector.data(), src_device, vector.size() * sizeof(Ty), cudaMemcpyDeviceToHost, stream));
}




template <typename Ty> void freeMem(Ty *dev_ptr) {
    /// safe free mem
    checkCudaStatus(cudaFreeAsync(dev_ptr, stream));
}

void memset(void *dev_ptr, int value, size_t size) {
    ///
    checkCudaStatus(cudaMemsetAsync(dev_ptr, value, size, stream));
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
void resetComputationContext(cudaError_t *error) {

    
}

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
    if (stream == NULL) {

        // implicit in utility::
        checkCudaStatus(cudaStreamCreate(&stream));

        int evID = 0;
        while (evID++ < CMAX) {
            // #observations
            checkCudaStatus(cudaEventCreate(prepStart));
            checkCudaStatus(cudaEventCreate(prepStop));
            checkCudaStatus(cudaEventCreate(computeStart));
            checkCudaStatus(cudaEventCreate(computeStop));
            checkCudaStatus(cudaEventCreate(solverStart));
            checkCudaStatus(cudaEventCreate(solverStop));
        }
    }
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




void ConstraintGetFullNorm(Computation *dev_ev) {

}


/// D.3.1.1. Device-Side Kernel Launch - kernel default shared memory, number of bytes
constexpr size_t Ns =  0;

/// grid size
constexpr size_t DIM_GRID = 1;

/// thread block size
constexpr size_t DIM_BLOCK = 1024;


/// lock for escaped data from computation rail
/// -- first conveged computation contex or last invalid

/// shared with cuda stream callback for wait, notify mechanism
std::condition_variable condition;

std::mutex mutex;

// host reference guarded by mutex 
std::atomic<Computation*> result;



/// <summary>
/// Computation Round Handler
/// </summary>
/// <param name="userData"></param>
void computationResultHandler(cudaStream_t stream, cudaError_t status, void *userData)
{
    // synchronize on stream
    
    Computation *computation = static_cast<Computation *>(userData);

    // obsluga bledow w strumieniu
    if (status != cudaSuccess) {

        const char *errorName = cudaGetErrorName(status);
        const char *errorStr = cudaGetErrorString(status);

        printf("[error] - computation id [%d] ,  %s = $s \n", computation->computationId, errorName, errorStr);
        return;
    }

    bool last = computation->computationId == (CMAX - 1);

    if (computation->norm2 < CONVERGENCE_LIMIT || last) {

        /// update result
        result.store(computation, std::memory_order_seq_cst);

        condition.notify_one();
    }

    // synchronize with stream next computation
}


///
/// Setup all matricies for computation and prepare kernel stream  intertwined with cuSolver
///
///
void solveSystemOnGPU(solver::SolverStat *stat, cudaError_t *error) {
    /// # Consideration -> ::"ingest stream and observe" fetch first converged
    ///
    //   -- napelniamy strumien i obserwujemy wektor bledy ASYNC , cudaStreamAddCallback( { stream, cudaError_t , userData } ):
    ///
    //   -- "cublas_v2.h" funkcje API czytaj/i wpisuja  wspolczynnik z HOST jak i z DEVICE
    ///
    //   -- to umozlwia asynchroniczna evaluacje
    ///
    //   -- zalaczamy zadanie FI(q) = 0 norm2 -> wpiszemy do ExecutionContext variable
    ///
    //   --- state-vector lineage memcpy(deviceToDevice)
    /// 

    
    int N = dimension;      

    /// prepare local result context
    result.store(NULL, std::memory_order_seq_cst);

    
    /// Uklad rownan liniowych  [ A * x = b ] powsta³y z linerazycji ukladu dynamicznego - tensory na urzadzeniu.
    double *dev_A = nullptr;
    double *dev_b = nullptr;
    double *dev_SV[CMAX] = {}; /// STATE VECTOR  -- lineage
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

    if (constraints.size() == 0 ) {
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
    printf("[ CUSOLVER ]  version ( Major.Minor.PatchLevel): %d.%d.%d\n", major, minor, patch);


    ///
    /// for [GPU] Evaluation  - dane obliczeniowego modelu !
    ///
    // immutables
    std::unique_ptr<Computation> ev(new Computation());
    ev->size = size;
    ev->coffSize = coffSize;
    ev->dimension = dimension = N;

    // immutables - 
    utility::memcpyToDevice(&d_points, points);
    utility::memcpyToDevice(&d_geometrics, geometrics);
    utility::memcpyToDevice(&d_constraints, constraints);
    utility::memcpyToDevice(&d_parameters, parameters);

    // immutables
    utility::memcpyToDevice(&d_pointOffset, pointOffset.get(), points.rbegin()->id);
    utility::memcpyToDevice(&d_parameterOffset, parameterOffset.get(), parameters.rbegin()->id);
    utility::memcpyToDevice(&d_accGeometricSize, accGeometricSize.get(), geometrics.size());
    utility::memcpyToDevice(&d_accConstraintSize, accConstraintSize.get(), constraints.size());


    checkCudaStatus(cudaStreamSynchronize(stream));

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
    /// one allocation block - slice

    utility::mallocToDevice(&dev_A, N * N * sizeof(double));
    utility::mallocToDevice(&dev_b, N * sizeof(double));
    utility::mallocToDevice(&dev_SV[0], N * sizeof(double)); // each computation seperate SV vector
    utility::mallocToDevice(&dev_dx, N * sizeof(double));


    checkCudaStatus(cudaStreamSynchronize(stream));

    ev->A = dev_A;
    ev->b = dev_b;
    ev->SV = dev_SV[0];
    ev->dx = dev_dx;

    ///
    /// [ GPU ] computation context mapped onto devive object
    /// 
    ///    
    // tu chce snapshot na kazda iteracje kopiowanie async, events [start,stop,workspace, streamId, stream]
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

    
    stat->startTime = graph::TimeNanosecondsNow();

    printf("@#=================== Solver Initialized ===================#@ \n");
    printf("");

    /// Inicjalizacja Macierzy A, b, i pochodne
      


/// SV - State Vector
    CopyIntoStateVector<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(dev_ev);

/// SV -> setup Lagrange multipliers    
    utility::memset(ev->SV + ev->size, 0, ev->coffSize);


/// # asynchronous kernel invocation

    checkCudaStatus(cudaStreamSynchronize(stream));



    linear_system_method_cuSolver_reset(stream);

    int itr = 0;
    
    // #########################################################################
    // 
    // idea - zapelniamy strumien w calym zakresie rozwiazania do CMAX
    // -- i zczytujemy pierwszy mozliwy poprawny wynik
    // 
    while (itr < CMAX) {

        // #observation
        checkCudaStatus(cudaEventRecord(computeStart[itr] ,stream));

        // #observation
        checkCudaStatus(cudaEventRecord(prepStart[itr], stream));
        
/// # KERNEL_PRE

        if (itr > 1){
            /// zerujemy macierz A      !!!!! second buffer
            utility::memset(dev_ev->A, 0, N * N * sizeof(double)); // --- ze wzgledu na addytywnosc
        }
        
/// macierz `A
                

        /// Cooficients Stiffnes Matrix
        ComputeStiffnessMatrix<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(dev_ev, geometrics.size());


#ifdef CDEBUG

        checkCudaStatus(cudaStreamSynchronize(stream));

        cudaError_t lastError = cudaPeekAtLastError();
        if (lastError != cudaSuccess) {
            printf(" [ kernel ] computation failed - stiffness matrix \n");
            throw std::logic_error("kernel - stiffness matrix");
        }
#endif

        /// 
        EvaluateConstraintHessian<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(dev_ev, N);

        /// upper Triangular
        /// --- (Wq); /// Jq = d(Fi)/dq --- write without intermediary matrix   `A = `A set value

        EvaluateConstraintJacobian<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(dev_ev, N);


        /// Tworzymy Macierz vector b vector `b
        
/// [ b ]  - right hand site

        /// Fr /// Sily  - F(q)
        EvaluateForceIntensity<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(dev_ev, N);     

        /// Fi / Wiezy  - Fi(q)
        EvaluateConstraintValue<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(dev_ev, N);          
        

        // [  b   ] ; /// --- schowac Do EvaluteForce - Constraint  .mulitply(-1);               


        ///  upper traingular

        // #observation
        checkCudaStatus(cudaEventRecord(prepStop[itr], stream));        


        // DENSE - CuSolver
        // LU Solver
        // 
/// ======== LINER SYSTEM equation CuSolver    === START
                
        // #observation
        checkCudaStatus(cudaEventRecord(solverStart[itr], stream));

        linear_system_method_cuSolver(dev_ev->A, dev_ev->b, dev_ev->dimension, stream);

        // #observation
        checkCudaStatus(cudaEventRecord(solverStop[itr], stream));

/// ======== LINER SYSTEM equation CuSolver    === STOP
      
        /// uaktualniamy punkty [ SV ] = [ SV ] + [ delta ]
        
        
        StateVectorAddDifference<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(ev.get()->SV, ev.get()->b, ev.get()->dimension);


        /// --- KERNEL_POST( __synchronize__ REDUCTION )


        // write into __device__ Computation block
        ConstraintGetFullNorm(dev_ev);        


        /// retrive ComputationFrom device

        void *userData = nullptr ;    // local_Computation

        /// callback dispatched to special cuda p_thread 
        checkCudaStatus(cudaStreamAddCallback(stream, computationResultHandler, userData, 0));              

        checkCudaStatus(cudaEventRecord(computeStop[itr], stream));

        itr++;
    }

/// all computation tiles submited - for first result , or invalid last

    std::unique_lock<std::mutex> ulock(mutex);

    /// atomic read
    if (result.load(std::memory_order_seq_cst) == nullptr) {

        /// spurious wakeup
        condition.wait(ulock, [] { return result.load(std::memory_order_seq_cst) != nullptr; });
    }
    
    // condition met
    ///
    /// 
    /// 
    /// 
    /// HOST computation view
    Computation *computation= result.load(std::memory_order_seq_cst);


    // STATE VECTOR result




    // oczekuje strumienia ale nie kopiujemy danych

    checkCudaStatus(cudaStreamSynchronize(stream));
    


    solverWatch.setStopTick();
    long solutionDelta = solverWatch.delta();

    printf("# solution delta : %d \n", solutionDelta); // print execution time
    printf("\n");                                      // print execution time

    CopyFromStateVector<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(computation);

    utility::memcpyFromDevice(points, d_points);

    stat->constraintDelta = computation->norm2;
    stat->convergence = computation->norm2 < CONVERGENCE_LIMIT;
    stat->stopTime = graph::TimeNanosecondsNow();
    stat->iterations = itr;
    stat->accSolverTime = 0;
    stat->accEvaluationTime = 0;
    stat->timeDelta = solverWatch.stopTick - solverWatch.startTick;

    /// w drugiej fazie dopisac => dodatkowo potrzebny per block  __shared__  memory   ORR    L1   fast region memory
    ///
    /// Evaluation data for  device  - CONST DATE for in process execution


    checkCudaStatus(cudaStreamSynchronize(stream));

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

        
    // implicit object for utility
    if (stream != NULL) {
        checkCudaStatus(cudaStreamDestroy(stream));
    }

    *error = cudaSuccess;
}

} // namespace solver


/// KERNEL#
void ConstraintEvaluateConstraintVector() {
    /// Wiezy  - Fi(q)
    /// b.mulitply(-1);
}



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