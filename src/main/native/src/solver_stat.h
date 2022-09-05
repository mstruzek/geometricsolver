#pragma once


namespace solver {

/// corespond to java implementations
struct SolverStat {
    SolverStat() = default;

    __int64 startTime;
    __int64 stopTime;
    __int64 timeDelta;
    size_t size;
    size_t coefficientArity;
    size_t dimension;
    __int64 accEvaluationTime;
    __int64 accSolverTime;
    bool convergence;
    double error;
    double constraintDelta;
    int iterations;
};

} // namespace solver