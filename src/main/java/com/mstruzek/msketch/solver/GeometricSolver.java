package com.mstruzek.msketch.solver;

public interface GeometricSolver {

    /**
     * Initialize or re-initialize solver state before this run.
     */
    void setup();

    /**
     * Statyczne modelowanie macierzy  na wewnetrzym statycznym modelu matematycznym.
     *
     * @return
     */
    SolverStat solveSystem(SolverStat solverStat);

}
