package com.mstruzek.msketch.solver;

public class SolverStat {

    /**
     * submit timestamp [ms]
     */
    public long startTime;

    /**
     * exit solver [ms]
     */
    public long stopTime;


    /** Total computation time [ns] */
    public long timeDelta;

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

}
