package com.mstruzek.msketch.solver;

public class SolverStat {

    /**
     * submit timestamp
     */
    public long startTime;

    /**
     * exit solver
     */
    public long stopTime;

    /**
     * solution space size (state vector)- 2x state vectors
     */
    public int size;

    /**
     * Lagrange coefficient arity.
     */
    public int coefficientArity;

    /**
     * solver matrix A dimension in linear equation
     */
    public int dimension;

    /**
     * Accumulated time for matrix evaluations [ms]
     */
    public long accEvaluationTime;

    /**
     * Accumulated time of internal linear equation solver [ms]
     */
    public long accSolverTime;

    /**
     * Solution converge below desired threshold
     */
    public boolean convergence;

    /**
     * Normalized error
     */
    public double error;

    /**
     * Constraint normalized delta
     */
    public double constraintDelta;

    /**
     * iterations
     */
    public int iterations;

    public void report(StateReporter reporter) {
        reporter.writeln("#=================== Time Space ===================#");
        reporter.writelnf("startTime         [ns]: %,20d ", startTime);
        reporter.writelnf("stopTime          [ns]: %,20d ", stopTime);
        reporter.writelnf("time elapsed      [ns]: %,20d ", (stopTime - startTime));
        reporter.writelnf("Acc Evaluation    [ns]: %,20d ", accEvaluationTime);
        reporter.writelnf("Acc Solver time   [ns]: %,20d ", accSolverTime);
        reporter.writeln("");

        reporter.writeln("#================== Solver space ==================#");
        reporter.writelnf("State vector dimension  : %d ", size);
        reporter.writelnf("Coefficients            : %d ", coefficientArity);
        reporter.writelnf("Matrix (A) dimension    : %s ", dimension + " x " + dimension);

        reporter.writeln("");
        reporter.writeln("#================== Error space ==================#");
        reporter.writelnf("convergence               : %s" , (convergence ? "T" : "F"));
        reporter.writelnf("`error                    : %e" , error);
        reporter.writelnf("constraint delta (error)  : %e" , constraintDelta);
        reporter.writelnf("iterations  (n)           : %d" , iterations);
        reporter.writeln("");
    }

}
