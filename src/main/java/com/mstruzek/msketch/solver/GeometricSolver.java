package com.mstruzek.msketch.solver;

public interface GeometricSolver {

    /**
     * initialize driver connection if necessary  !
     */
    void initializeDriver();


    /**
     * Initialize or re-initialize solver state before this run.
     */
    void setup();

    /**
     * Statyczne modelowanie macierzy  na wewnetrzym statycznym modelu matematycznym.
     *
     * @return
     */
    SolverStat solveSystem();

    /**
     * Close driver and free all resources
     */
    void destroyDriver();

}
