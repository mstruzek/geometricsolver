﻿#ifndef _GEOMETRIC_SOLVER_H_
#define _GEOMETRIC_SOLVER_H_

#include "cuda_runtime_api.h"

#include <memory>
#include <vector>

#include "quda.h" 
#include "model.cuh" 
#include "solver_stat.h"

#include "gpu_computation_context.h"
#include "gpu_computation.h"
#include "gpu_linear_system.h"




#define CONVERGENCE_LIMIT 10e-5

namespace solver {

class GPUComputation;

/*
 * initialize GPUComputationContext
 *
 * initialize for a single run exectly one GPUComputation
 *
 * - GPUComputation will borrow dev_ev,ev from computation context
 */
class GPUGeometricSolver {


  public:
    GPUGeometricSolver();

    ~GPUGeometricSolver();

    /**
     *
     */
    void registerPointType(int id, double px, double py);

    /**
     *
     */
    void registerGeometricType(int id, int geometricTypeId, int p1, int p2, int p3, int a, int b, int c, int d);

    /**
     *
     */
    void registerParameterType(int id, double value);

    /**
     *
     */
    void registerConstraintType(int id, int jconstraintTypeId, int k, int l, int m, int n, int paramId, double vecX,
                                double vecY);

    /**
    * 
    */
    void initComputation(cudaError_t *error);
    
    /**
    * 
    */
    void destroyComputation(cudaError_t *error);

    /**
     * @brief  setup all matricies for computation and prepare kernel stream  intertwined with cusolver
     *
     */
    void solveSystemOnGPU(SolverStat *stat, cudaError_t *error);


    std::shared_ptr<GPUComputation> getComputation();

  private:

    cudaStream_t stream = nullptr;


    utility::cu_vector<graph::Point> points;

    utility::cu_vector<graph::Geometric> geometrics;

    utility::cu_vector<graph::Constraint> constraints;

    utility::cu_vector<graph::Parameter> parameters;


    std::shared_ptr<GPUComputationContext> _computationContext;
    std::shared_ptr<solver::GPUComputation> _computation;    

    std::shared_ptr<GPULinearSystem> linearSystem;

    friend class GPUComputation;
};

} // namespace solver

#endif // _GEOMETRIC_SOLVER_H_