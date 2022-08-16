#ifndef _GEOMETRIC_SOLVER_H_
#define _GEOMETRIC_SOLVER_H_

#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

void checkCudaStatus_impl(cudaError_t status, size_t __line_);

#define checkCudaStatus(status) checkCudaStatus_impl(status, __LINE__)


namespace solver {

/// corespond to java implementations
struct SolverStat {
        SolverStat() = default;

        long startTime;
        long stopTime;
        long timeDelta;
        int size;
        int coefficientArity;
        int dimension;
        long accEvaluationTime;
        long accSolverTime;
        bool convergence;
        double error;
        double constraintDelta;
        int iterations;
};

/**
 *  reset all registers containing points, constraints, parameters !
 */
void resetComputationData(cudaError_t *error);

/**
 *  workspace - zwolnic pamiec i wyzerowac wskazniki , \\cusolver
 */
void resetComputationContext(cudaError_t *error);

/**
 * po zarejestrowaniu calego modelu w odpowiadajacych rejestrach , zainicjalizowac pomocnicze macierze
 *
 * przygotowac zmienne dla [cusolvera]
 *
 * przeliczenie pozycji absolutnej punktu na macierzy wyjsciowej
 */
void initComputationContext(cudaError_t *error);

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
void registerConstraintType(int id, int jconstraintTypeId, int k, int l, int m, int n, int paramId, double vecX, double vecY);

/**
 *
 */
double getPointPXCoordinate(int id);

/**
 *
 */
double getPointPYCoordinate(int id);

/**
 * @brief  setup all matricies for computation and prepare kernel stream  intertwined with cusolver
 *
 */
void solveSystemOnGPU(SolverStat *stat, cudaError_t *error);

                
void getPointCoordinateVector(double *state_arr);

} // namespace solver

#endif // _GEOMETRIC_SOLVER_H_