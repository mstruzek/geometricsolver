package com.mstruzek.msketch.solver;

import com.mstruzek.msketch.solver.jni.GpuGeometricSolverImpl;

public interface GeometricSolver {

    GeometricSolverType solverType();

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


    static GeometricSolver createInstance(GeometricSolverType solverType) {
        GeometricSolver solver = null;
        switch (solverType) {
            case CPU_SOLVER:
                solver = new GeometricSolverImpl();
                break;
            case GPU_SOLVER:
                solver = new GpuGeometricSolverImpl();
                break;
        };
        return solver;
    }

}
