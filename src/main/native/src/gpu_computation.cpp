#include "gpu_computation.h"

#include "cuda_runtime_api.h"

#include <functional>
#include <numeric>
#include <typeinfo>

#include <cusparse.h>

#include "model.cuh"
#include "solver_kernel.cuh"

#include "stop_watch.h"
#include "utility.cuh"

#ifdef DEBUG_GPU
#define validateStream validateStreamState()
#else
#define validateStream (void)(0)
#endif


namespace solver {

/// ====================

GPUComputation::GPUComputation(long computationId, cudaStream_t stream, std::shared_ptr<GPULinearSystem> linearSystem,
                               std::shared_ptr<GPUComputationContext> cc, GPUGeometricSolver *solver)
    : computationId(computationId), linearSystem(linearSystem), tensorOperation(stream), stream(stream),
      computationContext(cc) {

    /// HOST_PAGEABLE

    /// cuda_pinned_memory !

    /// change owner of compSt state
    points = std::move(solver->points);
    geometrics = std::move(solver->geometrics);
    constraints = std::move(solver->constraints);
    parameters = std::move(solver->parameters);

    if (points.size() == 0) {
        printf("empty solution space, add some geometric types\n");
        error = cudaSuccess;
        return;
    }

    // model aggreate for const immutable data
    if (points.size() == 0) {
        printf("[solver] - empty evaluation space model\n");
        error = cudaSuccess;
        return;
    }

    if (constraints.size() == 0) {
        printf("[solver] - no constraint configuration applied onto model\n");
        error = cudaSuccess;
        return;
    }

    this->computationMode = graph::getComputationMode(settings::get()->COMPUTATION_MODE);

    this->evaluationWatch.setStartTick();

    this->dev_SV = std::vector<double *>(CMAX, 0);

    /// setup all dependent structure for device , accSize or accOffset
    preInitializeData();

    /// prepera allocation and transfer structures onto device
    memcpyComputationToDevice();

    printf("[solver] model stage consistent !\n");
}

//////////////////===================================

void GPUComputation::preInitializeData() {

    /// mapping from point Id => point dest offset
    pointOffset = utility::stateOffset(points, [](auto point) { return point->id; });

    geometricOffset = utility::stateOffset(geometrics, [](auto geometric) { return geometric->id; });

    constraintOffset = utility::stateOffset(constraints, [](auto constraint) { return constraint->id; });

    /// this is mapping from Id => parameter dest offset
    parameterOffset = utility::stateOffset(parameters, [](auto parameter) { return parameter->id; });

    /// accumalted position of geometric block
    accGeometricSize = utility::accumulatedValue(geometrics, graph::geometricSetSize);

    /// accumulated position of constrain block
    accConstraintSize = utility::accumulatedValue(constraints, graph::constraintSize);

    if (computationMode == ComputationMode::SPARSE_LAYOUT) {
        /// accumulated COO - K
        accCooWriteStiffTensor = utility::accumulatedValue(geometrics, graph::tensorOpsCooStiffnesCoefficients);
        /// accumulated COO - Jacobian
        accCooWriteJacobianTensor = utility::accumulatedValue(constraints, graph::tensorOpsCooConstraintJacobian);

        // merge albo dwa buffory
    }

    /// `A` tensor internal structure dimensions
    size = std::accumulate(geometrics.begin(), geometrics.end(), 0,
                           [](auto acc, auto const &geometric) { return acc + graph::geometricSetSize(geometric); });

    coffSize = std::accumulate(constraints.begin(), constraints.end(), 0, [](auto acc, auto const &constraint) {
        return acc + graph::constraintSize(constraint);
    });

    dimension = size + coffSize;
}

void GPUComputation::memcpyComputationToDevice() {

    /// ============================================================
    ///         Host Computation with references to Device
    /// ============================================================

    /// const data in compSt
    utility::mallocAsync(&d_points, points.size(), stream);
    utility::mallocAsync(&d_geometrics, geometrics.size(), stream);
    utility::mallocAsync(&d_constraints, constraints.size(), stream);
    utility::mallocAsync(&d_parameters, parameters.size(), stream);

    utility::mallocAsync(&d_pointOffset, pointOffset.size(), stream);
    utility::mallocAsync(&d_geometricOffset, geometricOffset.size(), stream);
    utility::mallocAsync(&d_constraintOffset, constraintOffset.size(), stream);

    utility::mallocAsync(&d_parameterOffset, parameterOffset.size(), stream);

    utility::mallocAsync(&d_accGeometricSize, accGeometricSize.size(), stream);
    utility::mallocAsync(&d_accConstraintSize, accConstraintSize.size(), stream);

    if (computationMode == ComputationMode::SPARSE_LAYOUT) {
        utility::mallocAsync(&d_accCooWriteStiffTensor, accCooWriteStiffTensor.size(), stream);
        utility::mallocAsync(&d_accCooWriteJacobianTensor, accCooWriteJacobianTensor.size(), stream);

        /// option ` not-implemented-yet  +HESSIAN
        d_accCooWriteHessianTensor = NULL;

        cooWritesStiffSize = accCooWriteStiffTensor[accCooWriteStiffTensor.size() - 1];
        cooWirtesJacobianSize = accCooWriteJacobianTensor[accCooWriteJacobianTensor.size() - 1];
        int cooWritesSize = cooWritesStiffSize + cooWirtesJacobianSize * 2; /// +HESSIAN if computed

        // non-zero elements (coo/csr)
        nnz = cooWritesSize;

        /// sparse layout , first round only
        utility::mallocAsync(&d_cooRowInd, cooWritesSize, stream);
        utility::mallocAsync(&d_cooColInd, cooWritesSize, stream);
        utility::mallocAsync(&d_cooVal, cooWritesSize, stream);
        utility::mallocAsync(&d_PT, cooWritesSize, stream);
        utility::mallocAsync(&d_INV_PT, cooWritesSize, stream);

        utility::mallocAsync(&d_cooRowInd_tmp, cooWritesSize, stream);

        /// CSR

        /// rows = m + 1
        utility::mallocAsync(&d_csrRowPtrA, dimension + 1, stream);
        utility::mallocAsync(&d_csrColIndA, nnz, stream);
        utility::mallocAsync(&d_csrValA, nnz, stream);
    }

    /// ?// single  data-plane transfer

    // immutables -
    utility::memcpyAsync(&d_points, points, stream);
    utility::memcpyAsync(&d_geometrics, geometrics, stream);
    utility::memcpyAsync(&d_constraints, constraints, stream);
    utility::memcpyAsync(&d_parameters, parameters, stream);

    // immutables
    utility::memcpyAsync(&d_pointOffset, pointOffset.data(), pointOffset.size(), stream);
    utility::memcpyAsync(&d_geometricOffset, geometricOffset.data(), geometricOffset.size(), stream);
    utility::memcpyAsync(&d_constraintOffset, constraintOffset.data(), constraintOffset.size(), stream);

    utility::memcpyAsync(&d_accGeometricSize, accGeometricSize.data(), accGeometricSize.size(), stream);
    utility::memcpyAsync(&d_accConstraintSize, accConstraintSize.data(), accConstraintSize.size(), stream);

    if (!parameters.empty()) {
        utility::memcpyAsync(&d_parameterOffset, parameterOffset.data(), parameterOffset.size(), stream);
    }

    /// immutables
    if (computationMode == ComputationMode::SPARSE_LAYOUT) {
        /// accumulated COO
        utility::memcpyAsync(&d_accCooWriteStiffTensor, accCooWriteStiffTensor.data(), accCooWriteStiffTensor.size(),
                             stream);
        utility::memcpyAsync(&d_accCooWriteJacobianTensor, accCooWriteJacobianTensor.data(),
                             accCooWriteJacobianTensor.size(), stream);
    }

    const size_t N = dimension;
    ///
    ///  [ GPU ] tensors `A` `x` `dx` `b`  ------------------------
    ///
    utility::mallocAsync(&dev_A, N * N, stream);
    utility::mallocAsync(&dev_b, N, stream);
    utility::mallocAsync(&dev_dx, N, stream);

    for (int itr = 0; itr < CMAX; itr++) {
        /// each compSt data with its own StateVector
        utility::mallocAsync(&dev_SV[itr], N, stream);
    }
}

void GPUComputation::registerComputation(GPUComputation *computation) {
    GPUComputation::_registerComputation = computation;
}

void GPUComputation::unregisterComputation(GPUComputation *computation) { GPUComputation::_registerComputation = NULL; }

void GPUComputation::computationResultHandlerDelegate(cudaStream_t stream, cudaError_t status, void *userData) {
    if (_registerComputation) {
        _registerComputation->computationResultHandler(stream, status, userData);
    }
}

void GPUComputation::validateStreamState() {
    if (settings::get()->DEBUG_CHECK_ARG) {
        /// submitted kernel into  cuda driver
        checkCudaStatus(cudaPeekAtLastError());
        /// block and wait for execution
        checkCudaStatus(cudaStreamSynchronize(stream));
    }
}

void GPUComputation::PreInitializeComputationState(ComputationState *ev, int itr) {
    // compSt data;
    ev->cID = itr;
    ev->SV = dev_SV[itr]; /// data vector lineage
    ev->A = dev_A;
    ev->b = dev_b;
    ev->dx = dev_dx;
    ev->dev_norm = computationContext->get_dev_norm(itr);

    // geometric structure
    ev->size = size;
    ev->coffSize = coffSize;
    ev->dimension = dimension;

    ev->points = NVector<graph::Point>(d_points, points.size());
    ev->geometrics = NVector<graph::Geometric>(d_geometrics, geometrics.size());
    ev->constraints = NVector<graph::Constraint>(d_constraints, constraints.size());
    ev->parameters = NVector<graph::Parameter>(d_parameters, parameters.size());

    ev->pointOffset = d_pointOffset;
    ev->geometricOffset = d_geometricOffset;
    ev->constraintOffset = d_constraintOffset;
    ev->parameterOffset = d_parameterOffset;
    ev->accGeometricSize = d_accGeometricSize;
    ev->accConstraintSize = d_accConstraintSize;

    ev->computationMode = computationMode;

    /// =================================
    ///     Tensor A Computation Mode
    /// =================================
    if (computationMode == ComputationMode::SPARSE_LAYOUT) {

        ev->nnz = nnz;
        ev->cooWritesStiffSize = cooWritesStiffSize;
        ev->cooWirtesJacobianSize = cooWirtesJacobianSize;

        ev->accCooWriteStiffTensor = d_accCooWriteStiffTensor;
        ev->accCooWriteJacobianTensor = d_accCooWriteJacobianTensor;
        ev->accCooWriteHessianTensor = d_accCooWriteHessianTensor;

        ev->cooRowInd = d_cooRowInd; /// SparseLayout access
        ev->cooColInd = d_cooColInd; /// SparseLayout access
        ev->cooVal = d_cooVal;
    }
}

void GPUComputation::InitializeStateVector() {

    ///==================================================== ///
    ///            KERNEL ( Init State Vector )             ///
    ///==================================================== ///
    CopyIntoStateVector(stream, dev_SV[0], d_points, points.size());
    validateStream;

    ///==================================================== ///
    ///              Zreo Lagrange Coefficients             ///
    ///==================================================== ///
    utility::memsetAsync(dev_SV[0] + size, 0, coffSize, stream);
}

void GPUComputation::DeviceConstructTensorA(ComputationState *dev_ev, cudaStream_t stream) {

    /// Tensor A Sparse Memory Layout
    /// mem layout = [ Stiff + Jacobian + JacobianT  + Hessian*]

    ///==================================================== ///
    ///           KERNEL ( Stiff Tensor - K )               ///
    ///==================================================== ///
    ComputeStiffnessMatrix(stream, dev_ev, geometrics.size());
    validateStream;

    ///==================================================== ///
    ///           KERNEL ( Hessian Tensor - K )             ///
    ///==================================================== ///
    if (settings::get()->SOLVER_INC_HESSIAN) {
        EvaluateConstraintHessian(stream, dev_ev, constraints.size());
        validateStream;
    }

    /// Lower Tensor
    /// (Wq); /// Jq = d(Fi)/dq --- write without intermediary matrix
    ///

    ///==================================================== ///
    ///           KERNEL ( Lower Jacobian )                 ///
    ///==================================================== ///
    EvaluateConstraintJacobian(stream, dev_ev, constraints.size());
    validateStream;

    ///  Transposed Jacobian - Uperr Tensor
    ///  (Wq)';  Jq' = (d(Fi)/dq)' --- transposed - write without
    ///  intermediary matrix

    ///==================================================== ///
    ///           KERNEL ( Upper Jacobian )                 ///
    ///==================================================== ///
    EvaluateConstraintTRJacobian(stream, dev_ev, constraints.size());
    validateStream;
}

void GPUComputation::DeviceConstructTensorB(ComputationState *dev_ev, cudaStream_t stream) {
    ///
    /// B = [ SV ]  - right hand side
    ///
    ///==================================================== ///
    ///      KERNEL ( Point-Point-Tension F(q) )            ///
    ///==================================================== ///
    EvaluateForceIntensity(stream, dev_ev, geometrics.size());
    validateStream;

    ///==================================================== ///
    ///      KERNEL ( Constraint Tension Fi(q) )            ///
    ///==================================================== ///
    EvaluateConstraintValue(stream, dev_ev, constraints.size());
    validateStream;
}

void GPUComputation::DebugTensorConstruction(ComputationState *dev_ev) {
    if (settings::get()->DEBUG_TENSOR_A) {
        ///==================================================== ///
        ///      KERNEL ( stdout tensor at gpu )                ///
        ///==================================================== ///
        stdoutTensorData(stream, dev_ev, dimension, dimension);
        checkCudaStatus(cudaStreamSynchronize(stream));
    }

    if (settings::get()->DEBUG_TENSOR_B) {
        ///==================================================== ///
        ///      KERNEL ( stdout tensor at gpu )                ///
        ///==================================================== ///
        stdoutRightHandSide(stream, dev_ev, dimension);
        checkCudaStatus(cudaStreamSynchronize(stream));
    }
}

void GPUComputation::DebugTensorSV(ComputationState *dev_ev) {
    if (settings::get()->DEBUG_TENSOR_SV) {
        ///==================================================== ///
        ///      KERNEL ( stdout state vector  )                ///
        ///==================================================== ///
        stdoutStateVector(stream, dev_ev, dimension);
        checkCudaStatus(cudaStreamSynchronize(stream));
    }
    validateStream;
}

void GPUComputation::StateVectorAddDelta(double *dev_SV, double *dev_b) {

    ///==================================================== ///
    ///           KERNEL ( SV = SV + delta )                ///
    ///==================================================== ///
    StateVectorAddDifference(stream, dev_SV, dev_b, points.size());
    validateStream;

    // Uaktualniamy punkty SV = SV + delta
    // alternative : linearSystem->cublasAPIDaxpy(point_size, &alpha, dev_b, 1, dev_SV[itr], 1);
    // alternative : SV[i] = SV[i-1] + delta[i-1]
}

void GPUComputation::RebuildPointsFrom(ComputationState *computation) {

    ///==================================================== ///
    ///          KERNEL( Point From State Vector )          ///
    ///==================================================== ///
    CopyFromStateVector(stream, d_points, computation->SV, points.size());
    validateStream;

    utility::memcpyFromDevice(points, d_points, stream);
};

void GPUComputation::CudaGraphCaptureAndLaunch(cudaGraphExec_t graphExec) {

    /// Closing Stream Graph capturing mode

    cudaGraph_t graph;
    cudaGraphExecUpdateResult updateResult;
    cudaGraphNode_t errorNode;

    /// stream caputure capability
    checkCudaStatus(cudaStreamEndCapture(stream, &graph));

    if (graphExec != NULL) {
        // updateResult will store reason of failure
        cudaGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);
    }

    // First Execution if not initialize
    if (graphExec == NULL || updateResult != cudaGraphExecUpdateSuccess) {
        //
        if (graphExec != NULL) {
            checkCudaStatus(cudaGraphExecDestroy(graphExec));
        }
        const size_t bufferSize = 128;
        char pLogBuffer[bufferSize] = {};

        checkCudaStatus(cudaGraphInstantiate(&graphExec, graph, &errorNode, pLogBuffer, bufferSize));

        if (errorNode != NULL) {
            pLogBuffer[bufferSize - 1] = '\0';
            fprintf(stderr, "[error/graph] graph instantation error %s \n", pLogBuffer);
            exit(1);
        }
    }
    //
    checkCudaStatus(cudaGraphDestroy(graph));

    ///==================================================== ///
    ///                 Graph Execution                     ///
    ///==================================================== ///
    checkCudaStatus(cudaGraphLaunch(graphExec, stream));
}

GPUComputation *GPUComputation::_registerComputation = nullptr;

void GPUComputation::solveSystem(SolverStat *stat, cudaError_t *error) {

    // register C-function reference delegate - registerForDelegation(this) / unregisterFromDelegation(this)
    GPUComputation::registerComputation(this);

    size_t N = dimension;

    solverWatch.setStartTick();

    /// prepare local offset context
    result.store(NULL, std::memory_order_seq_cst);

    checkCudaStatus(cudaStreamSynchronize(stream));

    evaluationWatch.setStopTick();

    printf(" /////////////////// Solver Initialized /////////////////// \n\n");

    /// Graph Capturing Mechanism
    cudaGraphExec_t graphExec = NULL;

    int itr = 0;

#define FIRST_ROUND 0
#define NEXT_ROUND(v) v

    /// #########################################################################
    while (itr < CMAX) {
        /// preinitialize data vector

        computationContext->ComputeStart(itr);

        if (itr == FIRST_ROUND) {
            /// SV
            InitializeStateVector();

        } else {
            checkCudaStatus(
                cudaMemcpyAsync(dev_SV[itr], dev_SV[itr - 1], N * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        }

        ///  Host Context - with references to device
        ComputationState *ev = computationContext->host_ev(itr);
        ComputationState *dev_ev = computationContext->get_dev_ev(itr);

        /// snapshot transfer
        PreInitializeComputationState(ev, itr);

        utility::memcpyAsync(&dev_ev, ev, 1, stream);
        validateStream;

        computationContext->PrepStart(itr);

        /// zerujemy macierz A      !!!!! second buffer
        utility::memsetAsync(dev_A, 0, N * N, stream); // --- ze wzgledu na addytywnosc

        /// =============== Graph Capturing Mechanism ===============  //
        /// ## GRAPH_CAPTURE - BEGIN
        if (settings::get()->STREAM_CAPTURING) {

            /// stream caputure capability
            checkCudaStatus(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        }

        DeviceConstructTensorA(dev_ev, stream);

        DeviceConstructTensorB(dev_ev, stream);

        ///
        /// ##  GRAPH_CAPTURE - END
        /// =============== Graph Capturing Mechanism ===============  //
        if (settings::get()->STREAM_CAPTURING) {

            CudaGraphCaptureAndLaunch(graphExec);
        }

        if (computationMode == ComputationMode::SPARSE_LAYOUT) {
            /// COMPUTE in sorted CSR format

            const int m = static_cast<int>(dimension);
            const int n = static_cast<int>(dimension);

            if (itr == FIRST_ROUND) {

                /// COPYMK - for first iteration

                utility::memcpyOnDevice(d_cooRowInd_tmp, d_cooRowInd, nnz, stream);
                utility::memcpyOnDevice(d_csrColIndA, d_cooColInd, nnz, stream);


                ///  destructive operation - in-place  Xcoosort !!! requirment for solver and DirectLayout
                tensorOperation.convertToCsr(m, n, nnz, d_cooRowInd_tmp, d_csrColIndA, d_csrRowPtrA, d_PT);
                validateStream;

                /// gather d_csrValuA from cooValues;
                tensorOperation.gatherVector(nnz, CUDA_R_64F, d_cooVal, d_PT, d_csrValA);

                /// evaluate inverted permutation vector
                tensorOperation.invertPermuts(nnz, d_PT, d_INV_PT);

            } else { 
                
                /// ComputationMode::DIRECT_LAYOUT

                /// Inplace cooValus pivoting - async execution
                tensorOperation.gatherVector(nnz, CUDA_R_64F, d_cooVal, d_PT, d_csrValA);
                validateStream;
            }
        }
        ///
        computationContext->PrepStop(itr);

        DebugTensorConstruction(dev_ev);

#undef H_DEBUG

        double *host_norm = &ev->norm;
        //
        ///  ConstraintGetFullNorm
        //
        tensorOperation.vectorNorm(static_cast<int>(coffSize), (dev_b + size), host_norm);
        validateStream;

        computationContext->SolverStart(itr);

        if (computationMode == ComputationMode::DENSE_LAYOUT) {

            /// ------------------------------------------------------
            ///  Dense - CuSolver LINER SYSTEM equation solver
            /// ------------------------------------------------------

            linearSystem->solveLinearEquation(dev_A, dev_b, N);
            validateStream;

#ifdef H_DEBUG
            DebugTensorConstruction(dev_ev);
#endif
            /// uaktualniamy punkty SV = SV + delta
            StateVectorAddDelta(dev_SV[itr], dev_b);

        } else if (computationMode == ComputationMode::SPARSE_LAYOUT) {
            /// ------------------------------------------------------
            ///  Sparse - CuSolver LINER SYSTEM equation solver
            /// ------------------------------------------------------

            int const m = static_cast<int>(dimension);
            int const n = static_cast<int>(dimension);

            int* singularity = &ev->singularity;

            linearSystem->solverLinearEquationSP(m, n, nnz, d_csrRowPtrA, d_csrColIndA, d_csrValA, dev_b, dev_dx,
                                                 singularity);
            validateStream;

             if (settings::get()->DEBUG_CHECK_ARG) {
                checkCudaStatus(cudaStreamSynchronize(stream));
                 if (*singularity != -1) {
                     fprintf(stderr, "[solver] tensor A is not invertible at index = %d \n", *singularity);
                     ev->singularity = *singularity;
                     result.store(ev, std::memory_order_seq_cst);
                     break;
                 }
                 fprintf(stderr, "[solver] tensor A is invertible ;  %d \n", *singularity);
             }

             /// uaktualniamy state vector SV = SV + delta
             StateVectorAddDelta(dev_SV[itr], dev_dx);

        } else {
            fprintf(stderr, "[solver] computation mode not recognized ! %d \n", computationMode);
            exit(1);
        }             

        computationContext->SolverStop(itr);

        DebugTensorSV(dev_ev);

        /// ============================================================
        ///                 Set Computation Callback - Epsilon
        /// ============================================================
        AddComputationHandler(ev);

        computationContext->ComputeStop(itr);

        if (result.load(std::memory_order_seq_cst) != nullptr) {
            break;
        }

        /// Continue next ITERATION
        NEXT_ROUND(itr++);

    } // end_while
    /// #########################################################################

    /// Computation tiles submited
    std::unique_lock<std::mutex> ulock(mutex);

    /// Await last compSt state
    if (result.load(std::memory_order_seq_cst) == NULL) {
        /// spurious wakeup
        condition.wait(ulock, [&] { return result.load(std::memory_order_seq_cst) != NULL; });
    }

    ComputationState *compSt = (ComputationState *)result.load(std::memory_order_seq_cst);
    solverWatch.setStopTick();

    GPUComputation::unregisterComputation(this);

    checkCudaStatus(cudaStreamSynchronize(stream));

    if (settings::get()->DEBUG) {
        ///
        ///
        ReportComputationResult(compSt);
    }
    BuildSolverStat(compSt, stat);
    RebuildPointsFrom(compSt);

    if (graphExec != NULL) {
        cudaGraphExecDestroy(graphExec);
    }

    solverWatch.reset();
    evaluationWatch.reset();

    *error = cudaGetLastError();
}

void GPUComputation::AddComputationHandler(ComputationState *compState) {

    void *userData = static_cast<void *>(compState); // local_Computation
    checkCudaStatus(cudaStreamAddCallback(stream, GPUComputation::computationResultHandlerDelegate, userData, 0));
}

/// <summary>
/// Single Round Computation Handler
/// </summary>
/// <param name="userData"></param>
void GPUComputation::computationResultHandler(cudaStream_t stream, cudaError_t status, void *userData) {
    // synchronize on stream
    ComputationState *computation = static_cast<ComputationState *>(userData);

    // obsluga bledow w strumieniu
    if (status != cudaSuccess) {
        const char *errorName = cudaGetErrorName(status);
        const char *errorStr = cudaGetErrorString(status);
        printf("[error] - computation id [%d] ,  %s = %s \n", computation->cID, errorName, errorStr);
        return;
    }

    if (settings::get()->DEBUG) {
        fprintf(stdout, "[resutl/handler]- computationId (%d)  \n", computation->cID);
        fprintf(stdout, "[resutl/handler]- norm (%e) \n", computation->norm);
    }
    
    bool const last = computation->cID == (CMAX - 1);
    bool const isNan = isnan(computation->norm);
    bool const isNotConvertible = computation->singularity != -1;
    double const CONVERGENCE = settings::get()->CU_SOLVER_EPSILON;

    if (computation->norm < CONVERGENCE || isNan || last || isNotConvertible) {
        /// update offset
        auto expected = static_cast<ComputationState *>(nullptr);
        if (result.compare_exchange_strong(expected, computation)) {
            condition.notify_one();
        }
    }
    // synchronize with stream next compSt
}

void GPUComputation::BuildSolverStat(ComputationState *computation, solver::SolverStat *stat) {

    const double SOLVER_EPSILON = (settings::get()->CU_SOLVER_EPSILON);
    const int iter = computation->cID;

    stat->startTime = solverWatch.getStartTick();
    stat->stopTime = solverWatch.getStopTick();

    /// solverWatch.delta() + evaluationWatch.delta();
    stat->timeDelta = computationContext->getAccComputeTime(iter);

    stat->size = size;
    stat->coefficientArity = coffSize;
    stat->dimension = dimension;

    /// evaluationWatch.delta();
    stat->accEvaluationTime = computationContext->getAccPrepTime(iter);
    /// solverWatch.delta();
    stat->accSolverTime = computationContext->getAccSolverTime(iter);

    stat->convergence = computation->norm < SOLVER_EPSILON;
    stat->error = computation->norm;
    stat->constraintDelta = computation->norm;
    stat->iterations = computation->cID;
}

void GPUComputation::ReportComputationResult(ComputationState *computation) {
    long long solutionDelta = solverWatch.delta();

    printf("\n");
    printf("===================================================\n");
    printf("\n");
    printf("solution time delta [ns]        ( %zd )\n", solutionDelta);
    printf("\n");
    printf("===================================================\n");
    printf("comp ID                         ( %d )\n", computation->cID);
    printf("computation norm                ( %e )\n", computation->norm);
    printf("\n");
    printf("computation size                ( %zd )\n", computation->size);
    printf("computation coffSize            ( %zd )\n", computation->coffSize);
    printf("computation dimension           ( %zd )\n", computation->dimension);
    printf("\n");
    printf("===================================================\n");
    printf("\n");
}

double GPUComputation::getPointPXCoordinate(int id) {
    int offset = pointOffset[id];
    double px = points[offset].x;
    return px;
}

double GPUComputation::getPointPYCoordinate(int id) {
    int offset = pointOffset[id];
    double py = points[offset].y;
    return py;
}

void GPUComputation::fillPointCoordinateVector(double *stateVector) {
    graph::StopWatchAdapter stopWatch;
    stopWatch.setStartTick();
    // Effecient vector base loop  , time less then < 'unrolled 256,128, 64, .... pattern'
    for (size_t i = 0, size = points.size(); i < size; i++) {
        graph::Point &p = points[i];
        stateVector[2 * i] = p.x;
        stateVector[2 * i + 1] = p.y;
    }
    stopWatch.setStopTick();
    fprintf(stdout, "fillPointCoordinateVector delta: %llu ", stopWatch.delta());
}

int GPUComputation::updateConstraintState(int constraintId[], double vecX[], double vecY[], int size) {

    evaluationWatch.setStartTick();

    for (int itr = 0; itr < size; itr++) {
        int cId = constraintId[itr];
        int offset = constraintOffset[cId];
        graph::Constraint &constraint = constraints[offset];

        if (constraint.constraintTypeId != CONSTRAINT_TYPE_ID_FIX_POINT) {
            printf("[error] constraint type only supported is ConstraintFixPoint ! \n");
            return 1;
        }
        constraint.vecX = vecX[itr];
        constraint.vecY = vecY[itr];
    }

    if (d_constraints != nullptr) {
        utility::memcpyAsync(&d_constraints, constraints, stream);
    }
    return 0;
}

int GPUComputation::updateParametersValues(int parameterId[], double value[], int size) {
    for (int itr = 0; itr < size; itr++) {
        int pId = parameterId[itr];
        int offset = parameterOffset[pId];
        graph::Parameter &parameter = parameters[offset];
        parameter.value = value[itr];
    }

    if (d_parameters != nullptr) {
        utility::memcpyAsync(&d_parameters, parameters, stream);
    }
    return 0;
}

void GPUComputation::updatePointCoordinateVector(double stateVector[]) {
    for (size_t i = 0, size = points.size(); i < size; i++) {
        graph::Point &p = points[i];
        p.x = stateVector[2 * i];
        p.y = stateVector[2 * i + 1];
    }

    if (d_points != nullptr) {
        utility::memcpyAsync(&d_points, points, stream);
    }
}

//////////////////===================================

GPUComputation::~GPUComputation() {
    /// at least one solver compSt

    for (int itr = 0; itr < CMAX; itr++) {
        utility::freeAsync(dev_SV[itr], stream);
    }

    utility::freeAsync(dev_A, stream);
    utility::freeAsync(dev_b, stream);
    utility::freeAsync(dev_dx, stream);

    utility::freeAsync(d_parameterOffset, stream);
    utility::freeAsync(d_accGeometricSize, stream);
    utility::freeAsync(d_accConstraintSize, stream);
    utility::freeAsync(d_constraintOffset, stream);
    utility::freeAsync(d_geometricOffset, stream);
    utility::freeAsync(d_pointOffset, stream);

    utility::freeAsync(d_points, stream);
    utility::freeAsync(d_geometrics, stream);
    utility::freeAsync(d_constraints, stream);
    utility::freeAsync(d_parameters, stream);

    utility::freeAsync(d_accCooWriteStiffTensor, stream);
    utility::freeAsync(d_accCooWriteJacobianTensor, stream);

    utility::freeAsync(d_cooRowInd, stream);
    utility::freeAsync(d_cooColInd, stream);
    utility::freeAsync(d_cooVal, stream);
    utility::freeAsync(d_PT, stream);
    utility::freeAsync(d_INV_PT, stream);

    utility::freeAsync(d_cooRowInd_tmp, stream);

    /// CSR
    utility::freeAsync(d_csrRowPtrA, stream);
    utility::freeAsync(d_csrColIndA, stream);
    utility::freeAsync(d_csrValA, stream);
}

} // namespace solver

#undef validateStream