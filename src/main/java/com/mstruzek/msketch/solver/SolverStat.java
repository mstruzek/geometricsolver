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
    public boolean converged;

    /**
     * Normalized error
     */
    public double delta;

    /**
     * Constraint normalized delta
     */
    public double constraintDelta;

    /**
     * iterations
     */
    public int iterations;

    public void report(StateReporter reporter) {
        reporter.writeln("startTime     [ ms ] : " + startTime);
        reporter.writeln("stopTime      [ ms ] : " + stopTime);
        reporter.writeln("time elapsed  [ ms ] : " + (stopTime - startTime));
        reporter.writeln("time elapsed  [ ms ] : " + (stopTime - startTime));

        reporter.writeln("");

        reporter.writeln("=== Solver space");
        reporter.writeln("state vector size     : " + size);
        reporter.writeln("coefficients          : " + coefficientArity);
        reporter.writeln("acc evaluation  [ms]  : " + accEvaluationTime);
        reporter.writeln("matrix A dimension    : " + dimension + " x " + dimension);
        reporter.writeln("acc solver time [ms]  : " + accSolverTime);

        reporter.writeln("");
        reporter.writeln("converged                 : " + (converged ? "T" : "F"));
        reporter.writeln("delta (error)             : " + delta);
        reporter.writeln("constraint delta (error)  : " + constraintDelta);
        reporter.writeln("iterations  (n)           : " + iterations);
        reporter.writeln("");
    }

}
