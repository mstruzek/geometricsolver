#ifndef _GEOMETRIC_SOLVER_H_
#define _GEOMETRIC_SOLVER_H_

#include <memory>
#include <vector>

#include "cuda_runtime.h"




namespace solver {

/// corespond to java implementations
struct SolverStat {
        SolverStat() = default;

        long long startTime;
        long long stopTime;
        long long timeDelta;
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
 * 
 */
void initComputationContext(cudaError_t *error);

/**
 *  reset all registers containing points, constraints, parameters !
 */
void destroyComputation(cudaError_t *error);

/**
 *  workspace - zwolnic pamiec i wyzerowac wskazniki , \\cusolver
 */
void destroyComputationContext(cudaError_t *error);

/**
 * po zarejestrowaniu calego modelu w odpowiadajacych rejestrach , zainicjalizowac pomocnicze macierze
 *
 * przygotowac zmienne dla [cusolvera]
 *
 * przeliczenie pozycji absolutnej punktu na macierzy wyjsciowej
 * 
 * commitTime --
 */
void initComputation(cudaError_t *error);


/*
 * Last commit of structural changes applied into model.
 */
long getCommitTime();

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
                
/**
 * fetch current state vector from last computation context
 */
void fillPointCoordinateVector(double *stateVector);

/**
 * update point coordinates after modifications in java
 */
void updatePointCoordinateVector(double *stateVector);

/**
 *  update constraint fixed vectors 
 */
int updateConstraintState(int *constraintId, double  *vecX, double *vecY, int size);


int updateParametersValues(int *parameterId, double *value, int size);

} // namespace solver

#endif // _GEOMETRIC_SOLVER_H_