#pragma once


namespace solver {

/// corespond to java implementations
struct SolverStat {
    SolverStat() = default;

    long long startTime;
    long long stopTime;
    long long timeDelta;
    size_t size;
    size_t coefficientArity;
    size_t dimension;
    long long accEvaluationTime;
    long long accSolverTime;
    bool convergence;
    double error;
    double constraintDelta;
    int iterations;
};

} // namespace solver