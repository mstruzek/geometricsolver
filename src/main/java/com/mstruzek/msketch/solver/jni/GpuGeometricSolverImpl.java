package com.mstruzek.msketch.solver.jni;

import com.mstruzek.jni.JNIDebugCode;
import com.mstruzek.jni.JNISolverGate;
import com.mstruzek.msketch.*;
import com.mstruzek.msketch.solver.GeometricSolver;
import com.mstruzek.msketch.solver.GeometricSolverType;
import com.mstruzek.msketch.solver.SolverStat;
import com.mstruzek.msketch.solver.StateReporter;

import java.util.LinkedList;
import java.util.List;

import static com.mstruzek.jni.JNIDebugCode.Decision.NO;
import static com.mstruzek.jni.JNIDebugCode.Decision.YES;
import static java.util.stream.Collectors.toCollection;

public class GpuGeometricSolverImpl implements GeometricSolver {

    private StateReporter reporter;

    private long lastSnapshotId = Long.MIN_VALUE;

    @Override
    public GeometricSolverType solverType() {
        return GeometricSolverType.GPU_SOLVER;
    }

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

//        JNISolverGate.setBooleanProperty(JNIDebugCode.DEBUG.code, false);

        JNISolverGate.setBooleanProperty(JNIDebugCode.STREAM_CAPTURING.code, YES);


        JNISolverGate.setDoubleProperty(JNIDebugCode.CU_SOLVER_EPSILON.code, 10e-2);
        JNISolverGate.setLongProperty(JNIDebugCode.GRID_SIZE.code, 4); // NOT_USED
        JNISolverGate.setLongProperty(JNIDebugCode.BLOCK_SIZE.code, 512);

        JNISolverGate.setBooleanProperty(JNIDebugCode.DEBUG_CHECK_ARG.code, NO);
        JNISolverGate.setBooleanProperty(JNIDebugCode.DEBUG_TENSOR_A.code, NO);
        JNISolverGate.setBooleanProperty(JNIDebugCode.SOLVER_INC_HESSIAN.code, NO);

//        JNISolverGate.setBooleanProperty(JNIDebugCode.DEBUG_TENSOR_B.code, false);
//        JNISolverGate.setBooleanProperty(JNIDebugCode.DEBUG_TENSOR_SV.code, true);
//        JNISolverGate.setBooleanProperty(JNIDebugCode.DEBUG_SOLVER_CONVERGENCE.code, YES);

    }

    @Override
    public SolverStat solveSystem() {

        int err = 0; /// JNI_SUCCESS

        if (ModelRegistry.constraintCounter == 0) {
            reporter.writeln("[solver/gpu] no constraints registered on model");
            return null;
        }

        long nextComputation = ModelRegistry.computationSnapshotId();
        reporter.writelnf("[solver/gpu]  ! snapshot id == %d" , nextComputation);
        if (lastSnapshotId != nextComputation) {
            /*
             *   register model after structural changes
             */
            err = JNISolverGate.destroyComputation();
            if (err != JNISolverGate.JNI_SUCCESS) {
                reporter.writeln("[solver/gpu] destroy computation error !");
                return null;
            }

            boolean registered = registerModelOnGPU();

            if (!registered) {
                reporter.writeln("[solver/gpu] model registration error !");
                return null;
            }


            reporter.writeln("[solver/gpu] model registration OK !");

            err = JNISolverGate.initComputation();

            if (err != JNISolverGate.JNI_SUCCESS) {
                reporter.writeln("[solver/gpu] model computation data initialization error !");
                return null;
            }

            lastSnapshotId = nextComputation;

        } else {
            /// positional changes
            updateConstraintsState();
            ///
            updateParametersValues();
            ///
            updateStateVector();
        }

        /*
         * -----------------  SOLVER -------------------
         */

        try {

            err = JNISolverGate.solveSystem();

        } catch (Exception e) {
            e.printStackTrace();
            System.out.flush();
        }


        if (err != JNISolverGate.JNI_SUCCESS) {
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


    private void updateConstraintsState() {

        final List<ConstraintFixPoint> constraintsFixed = ModelRegistry.dbConstraint().values().stream()
            .filter(constraint -> constraint.getConstraintType().equals(ConstraintType.FixPoint))
            .map(ConstraintFixPoint.class::cast)
            .collect(toCollection(LinkedList::new));

        final int size = constraintsFixed.size();

        int[] constraintId = new int[size];
        double[] vecX = new double[size];
        double[] vecY = new double[size];

        int itr = 0;
        for (ConstraintFixPoint constraint : constraintsFixed) {
            Vector fixVector = constraint.getFixVector();
            constraintId[itr] = constraint.getConstraintId();
            vecX[itr] = fixVector.getX();
            vecY[itr] = fixVector.getY();
            itr++;
        }
        JNISolverGate.updateConstraintState(constraintId, vecX, vecY, size);
    }

    private void updateStateVector() {
        double[] stateVector = new double[2 * ModelRegistry.dbPoint().size()];

        int itr = 0;
        for (int pointId : ModelRegistry.dbPoint().keySet()) {
            Point p = ModelRegistry.dbPoint().get(pointId);
            stateVector[2 * itr] = p.getX();
            stateVector[2 * itr + 1] = p.getY();
            itr++;
        }

        /// JNI commit
        JNISolverGate.updateStateVector(stateVector);
    }

    private void updateParametersValues() {
        final int size = ModelRegistry.dbParameter().size();
        double[] value = new double[size];
        int[] parameterId = new int[size];
        int itr = 0;
        for (Parameter parameter : ModelRegistry.dbParameter().values()) {
            parameterId[itr] = parameter.getId();
            value[itr] = parameter.getValue();
            itr++;
        }
        JNISolverGate.updateParametersValues(parameterId, value, size);
    }

    @Override
    public void destroyDriver() {

        int error = JNISolverGate.destroyComputation();
        if (error != JNISolverGate.JNI_SUCCESS) {
            reporter.writelnf("[gpu/solver] destroy computation error !");
        }

        error = JNISolverGate.destroyComputationContext();
        if (error != JNISolverGate.JNI_SUCCESS) {
            reporter.writelnf("[gpu/solver] destroy computation context error !");
        }

        error = JNISolverGate.closeDriver();
        if (error != JNISolverGate.JNI_SUCCESS) {
            reporter.writelnf("[gpu/solver] close driver error !");
        }
    }

    /**
     * Fetch all computed points according to state vector ordering by Point[ID] property.
     *
     * Data in moved from State Vector of gpu geometric solver.
     */
    private void fetchGPUComputedPositionsIntoModel() {
        double[] coordinateVector = JNISolverGate.fetchStateVector();
        int itr = 0;
        for (int pointId : ModelRegistry.dbPoint().keySet()) {
            double px = coordinateVector[itr * 2];
            double py = coordinateVector[itr * 2 + 1];
            ModelRegistry.dbPoint().get(pointId).Vector().setLocation(px, py);
            itr++;
        }
        /*
         *  double px = JNISolverGate.getPointPXCoordinate(pointId);
         *  double py = JNISolverGate.getPointPYCoordinate(pointId);
         *
         */
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
