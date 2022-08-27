package com.mstruzek.msketch.solver.jni;

import com.mstruzek.jni.JNIDebugCode;
import com.mstruzek.jni.JNISolverGate;
import com.mstruzek.msketch.*;
import com.mstruzek.msketch.solver.GeometricSolver;
import com.mstruzek.msketch.solver.SolverStat;
import com.mstruzek.msketch.solver.StateReporter;

public class GpuGeometricSolverImpl implements GeometricSolver {

    private StateReporter reporter;

    private long lastCommitTime = 0;

    private long lastSnapshotId = Long.MIN_VALUE;

    @Override
    public void initializeDriver() {

        StateReporter.DebugEnabled = false;

        reporter = StateReporter.getInstance();

        int error = JNISolverGate.initDriver(0);

        if (error != JNISolverGate.JNI_SUCCESS) {
            reporter.writeln(" [GPU] driver failed - inspect jnigsketcher.so or *.dll file !");
            return;
        }

        error = JNISolverGate.initComputationContext();
        if (error != JNISolverGate.JNI_SUCCESS) {
            reporter.writeln(" [GPU] failed computation context initializer!");
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

        JNISolverGate.setBooleanProperty(JNIDebugCode.DEBUG.code, true);
        JNISolverGate.setBooleanProperty(JNIDebugCode.DEBUG_TENSOR_A.code, true);
        JNISolverGate.setBooleanProperty(JNIDebugCode.DEBUG_TENSOR_B.code, true);
        JNISolverGate.setBooleanProperty(JNIDebugCode.DEBUG_TENSOR_SV.code, true);
        JNISolverGate.setBooleanProperty(JNIDebugCode.DEBUG_SOLVER_CONVERGENCE.code, true);
    }

    @Override
    public SolverStat solveSystem(SolverStat solverStat) {

        int err = 0; /// JNI_SUCCESS

        if (ModelRegistry.constraintCounter == 0) {
            reporter.writeln("[solver/gpu] no constraints registered on model");
            return null;
        }

        if(lastSnapshotId != computationSnapshotId()) {
            /*
             *   REGISTER MODEL
             */
            reporter.writeln("--------------------------------");

            err = JNISolverGate.destroyComputation();
            if(err != JNISolverGate.JNI_SUCCESS) {
                reporter.writeln("[solver/gpu] destroy computation error !");
                return null;
            }

            boolean registered = registerModelOnGPU();

            if(!registered) {
                reporter.writeln("[solver/gpu] model registration error !");
                return null;
            }


            reporter.writeln("[solver/gpu] model registration OK !");

            err = JNISolverGate.initComputation();

            if(err != JNISolverGate.JNI_SUCCESS) {
                reporter.writeln("[solver/gpu] model computation data initialization error !");
                return null;
            }

            lastSnapshotId = computationSnapshotId();

        } else {
            /// positional changes
            /// updateConstraintState(id, vecX, vecY) - ConstraintFixPoint for fixed control points

            updateStateVector();
        }

        /*
         * -----------------  SOLVER -------------------
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

    private void updateStateVector() {
        double[] stateVector = new double[2 * ModelRegistry.dbPoint().size()];

        int itr = 0;
        for (int pointId: ModelRegistry.dbPoint().keySet()) {
            Point p = ModelRegistry.dbPoint().get(pointId);
            stateVector[2 * itr  ] = p.getX();
            stateVector[2 * itr + 1 ] = p.getY();
            itr++;
        }

        JNISolverGate.updateStateVector(stateVector);
    }

    @Override
    public void destroyDriver() {

        int error = JNISolverGate.destroyComputation();
        if(error != JNISolverGate.JNI_SUCCESS) {
            reporter.writelnf("[gpu/solver] destroy computation error !");
        }

        error = JNISolverGate.destroyComputationContext();
        if(error != JNISolverGate.JNI_SUCCESS) {
            reporter.writelnf("[gpu/solver] destroy computation context error !");
        }

        error = JNISolverGate.closeDriver();
        if(error != JNISolverGate.JNI_SUCCESS) {
            reporter.writelnf("[gpu/solver] close driver error !");
        }
   }

    /**
     * Fetch all computed points according to state vector ordering by Point[ID] property.
     *
     * Data in moved from State Vector of gpu geometric solver.
     */
    private void fetchGPUComputedPositionsIntoModel() {
        //double[] coordinateVector = JNISolverGate.fetchStateVector();
        int itr = 0;
        for (int pointId: ModelRegistry.dbPoint().keySet()) {
/*
            double px = coordinateVector[ itr * 2 ];
            double py = coordinateVector[ itr * 2 + 1];
*/
            double px = JNISolverGate.getPointPXCoordinate(pointId);
            double py = JNISolverGate.getPointPYCoordinate(pointId);
            ModelRegistry.dbPoint().get(pointId).Vector().setLocation(px, py);
            itr++;
        }
    }

    private long computationSnapshotId() {
        long hash = Long.MIN_VALUE;
        for (Point point : ModelRegistry.dbPoint().values())
            hash += point.getId();
        for (GeometricObject geometric : ModelRegistry.dbPrimitives().values())
            hash += geometric.getPrimitiveId() * 2L;
        for (Parameter parameter : ModelRegistry.dbParameter().values())
            hash += parameter.getId() * 4L;
        for (Constraint constraint : ModelRegistry.dbConstraint().values())
            hash += constraint.getConstraintId() * 3L;

        return hash;
    }

    /**
     * Register current database model in GPU solver context.
     */
    private boolean registerModelOnGPU() {

        /// read/write current state hash function from id

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
