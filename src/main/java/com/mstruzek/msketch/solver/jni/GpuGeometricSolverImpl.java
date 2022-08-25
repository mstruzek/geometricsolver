package com.mstruzek.msketch.solver.jni;

import com.mstruzek.jni.JNISolverGate;
import com.mstruzek.msketch.*;
import com.mstruzek.msketch.solver.GeometricSolver;
import com.mstruzek.msketch.solver.SolverStat;
import com.mstruzek.msketch.solver.StateReporter;

public class GpuGeometricSolverImpl implements GeometricSolver {

    private StateReporter reporter;

    @Override
    public void initializeDriver() {

        StateReporter.DebugEnabled = true;

        reporter = StateReporter.getInstance();

        int error = JNISolverGate.initDriver();

        if (error != JNISolverGate.JNI_SUCCESS) {
            reporter.writeln(" [GPU] driver failed - inspect jnigsketcher.so or *.dll file !");
            return;
        }

        /*
         *
         * Otherwise, drive establish connection with device id 1 - const.
         */

        reporter.writeln(" [ GPU ] driver connection success");
    }

    @Override
    public void setup() {

    }

    @Override
    public SolverStat solveSystem(SolverStat solverStat) {

        int err = 0; /// JNI_SUCCESS

        if (ModelRegistry.constraintCounter == 0) {
            reporter.writeln("[solver/gpu] no constraints registered on model");
            return null;
        }

        /// ?????

        err = JNISolverGate.resetComputationContext();

        if (err != JNISolverGate.JNI_SUCCESS) {
            reporter.writeln("[solver/gpu] reset computation Context operation failed !");
            return null;
        }

        err = JNISolverGate.resetComputationData();

        if (err != JNISolverGate.JNI_SUCCESS) {
            reporter.writeln("[solver/gpu] reset computation Data operation failed !");
            return null;
        }

        /*
         * ===================== REGISTER MODEL ======================
         */
        reporter.writeln("=========================================");

        boolean registered = registerModelOnGPU();

        if(!registered) {
            reporter.writeln("[solver/gpu] model registration failed !");
            return null;
        }
        reporter.writeln("[solver/gpu] model registration OK !");

        err = JNISolverGate.initComputationContext();

        if(err != JNISolverGate.JNI_SUCCESS) {
            reporter.writeln("[solver/gpu] model computation context initialization failed !");
            return null;
        }

        /*
         * ======================== SOLVER ======================
         */
        err = JNISolverGate.solveSystem();


        if(err!= JNISolverGate.JNI_SUCCESS) {
            reporter.writelnf("[solver/gpu] solver execution failed with error = %s!", JNISolverGate.getLastError());
            return null;
        }

        final SolverStat solverStatistics = JNISolverGate.getSolverStatistics();

        reporter.writelnf("[solver/gpu] computation success  !");
        reporter.writelnf(" -- computation lastError    : %s", JNISolverGate.getLastError());
        reporter.writelnf(" -- computation convergence  : %b", solverStatistics.convergence);

        fetchGPUComputedPositionsIntoModel();

        return solverStatistics;
    }

    /**
     * Fetch all computed points according to state vector ordering by Point[ID] property.
     *
     * Data in moved from State Vector of gpu geometric solver.
     */
    private void fetchGPUComputedPositionsIntoModel() {

//        double[] coordinateVector = JNISolverGate.getPointCoordinateVector();

        for (int pointId: ModelRegistry.dbPoint().keySet()) {
/*
            double px = coordinateVector[ pointId * 2 ];
            double py = coordinateVector[ pointId * 2 + 1];
*/

            double px = JNISolverGate.getPointPXCoordinate(pointId);
            double py = JNISolverGate.getPointPYCoordinate(pointId);

            ModelRegistry.dbPoint().get(pointId).Vector().setLocation(px, py);
        }
    }


    /**
     * Register current database model in GPU solver context.
     */
    private boolean registerModelOnGPU() {

        // register point set
        for (Point point : ModelRegistry.dbPoint().values()) {
            int id = point.getId();
            double px = point.getX();
            double py = point.getY();
            int err = JNISolverGate.registerPointType(id, px, py);
            if (err != JNISolverGate.JNI_SUCCESS) {
                reporter.writelnf("[solver/gpu] point registration failed : %s", point.toString());
                return false;
            }
        }

        // register geometric objects
        for (GeometricObject geometric : ModelRegistry.dbPrimitives().values()) {
            int primitiveId = geometric.getPrimitiveId();
            int geometricType = geometric.getType().ordinal();
            int p1 = geometric.getP1();
            int p2 = geometric.getP2();
            int p3 = geometric.getP3();
            int a = geometric.getA();
            int b = geometric.getB();
            int c = geometric.getC();
            int d = geometric.getD();
            int err = JNISolverGate.registerGeometricType(primitiveId, geometricType, p1, p2, p3, a, b, c, d);
            if (err != JNISolverGate.JNI_SUCCESS) {
                reporter.writelnf("[solver/gpu] point registration failed : %s", geometric.toString());
                return false;
            }
        }

        // register parameter set
        for (Parameter parameter : ModelRegistry.dbParameter().values()) {
            int parameterId = parameter.getId();
            double value = parameter.getValue();
            int err = JNISolverGate.registerParameterType(parameterId, value);
            if (err != JNISolverGate.JNI_SUCCESS) {
                reporter.writelnf("[solver/gpu] parameter registration failed : %s", parameter.toString());
                return false;
            }
        }

        // register constraint set
        for (Constraint constraint : ModelRegistry.dbConstraint().values()) {
            int constraintId = constraint.getConstraintId();
            int constraintTypeId = constraint.getConstraintType().ordinal();
            int k = constraint.getK();
            int l = constraint.getL();
            int m = constraint.getM();
            int n = constraint.getN();
            int paramId = constraint.getParameter();
            double vecX = 0.0;
            double vecY = 0.0;
            if (constraint instanceof ConstraintFixPoint) {
                ConstraintFixPoint fixPoint = (ConstraintFixPoint) constraint;
                Vector fixVector = fixPoint.getFixVector();
                vecX = fixVector.getX();
                vecY = fixVector.getY();
            }
            int err = JNISolverGate.registerConstraintType(constraintId, constraintTypeId, k, l, m, n, paramId, vecX, vecY);
            if (err != JNISolverGate.JNI_SUCCESS) {
                reporter.writelnf("[solver/gpu] constraint registration failed : %s", constraint.toString());
                return false;
            }
        }
        /// Model registered properly without conflicts !
        return true;
    }

}
