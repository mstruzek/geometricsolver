package com.mstruzek.msketch.solver;

public interface GeometricSolver {

    /**
     * Statyczne modelowanie macierzy  na wewnetrzym statycznym modelu matematycznym.
     *
     * @return
     */
    SolverStat solveSystem(SolverStat solverStat);

}
