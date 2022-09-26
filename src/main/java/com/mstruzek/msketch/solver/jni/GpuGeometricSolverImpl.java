package com.mstruzek.msketch.solver.jni;

import com.mstruzek.jni.JNIDebugParameters;
import com.mstruzek.jni.JNIDebugParameters.SolverMode;
import com.mstruzek.jni.JNISolverGate;
import com.mstruzek.msketch.*;
import com.mstruzek.msketch.solver.GeometricSolver;
import com.mstruzek.msketch.solver.GeometricSolverType;
import com.mstruzek.msketch.solver.SolverStat;
import com.mstruzek.msketch.solver.StateReporter;

import java.util.LinkedList;
import java.util.List;

import static com.mstruzek.jni.JNIDebugParameters.ComputationMode.*;
import static com.mstruzek.jni.JNISolverGate.JNI_SUCCESS;
import static java.util.stream.Collectors.toCollection;

public class GpuGeometricSolverImpl implements GeometricSolver {

    private StateReporter reporter;

    private long lastSnapshotId = Long.MIN_VALUE;

    private static final int DEVICE_ID = 0;

    @Override
    public GeometricSolverType solverType() {
        return GeometricSolverType.GPU_SOLVER;
    }

    @Override
    public void initializeDriver() {

        StateReporter.DebugEnabled = false;
        reporter = StateReporter.getInstance();

        int error = initDriver(DEVICE_ID);
        if (error != JNI_SUCCESS) {
            reporter.writeln(" [GPU] driver failed - inspect jnigsketcher.so or *.dll file !");
            return;
        }
        error = initComputationContext();
        if (error != JNI_SUCCESS) {
            reporter.writeln(" [GPU] failed computation context initializer!");
            return;
        }

        reporter.writeln(" [ GPU ] driver connection success");
    }

    /**
     * QR decomposition - solver convergence  , STABILITY!
     */
    @Override
    public void setup() {
//       JNIDebugParameters.DEBUG.setBooleanProperty(false);
        JNIDebugParameters.STREAM_CAPTURING.setBooleanProperty(false);
        JNIDebugParameters.DEBUG_CSR_FORMAT.setBooleanProperty(false);
        /// OK :
        JNIDebugParameters.COMPUTATION_MODE.setLongProperty(COMPACT_MODE);
        // JNIDebugParameters.COMPUTATION_MODE.setLongProperty(DIRECT_MODE);
//        JNIDebugParameters.SOLVER_MODE.setLongProperty(SolverMode.QR_SPARSE);
        JNIDebugParameters.SOLVER_MODE.setLongProperty(SolverMode.ILU_BiCG_STAB);
        //        JNIDebugParameters.COMPUTATION_MODE.setLongProperty(SPARSE_LAYOUT);
//        JNIDebugParameters.COMPUTATION_MODE.setLongProperty(DIRECT_LAYOUT);

        JNIDebugParameters.SOLVER_EPSILON.setDoubleProperty(1e-5);
        /// synchronize cuda stream
        JNIDebugParameters.DEBUG_CHECK_ARG.setBooleanProperty(false);
        JNIDebugParameters.SOLVER_INC_HESSIAN.setBooleanProperty(false);
        JNIDebugParameters.DEBUG_TENSOR_SV.setBooleanProperty(false);

//        JNIDebugParameters.GRID_SIZE.setLongProperty(4); // NOT_USED
//        JNIDebugParameters.BLOCK_SIZE.setLongProperty(512);
//        JNIDebugParameters.DEBUG_TENSOR_A.setBooleanProperty(false);
//        JNIDebugParameters.DEBUG_TENSOR_B.setBooleanProperty(false);
//        JNIDebugParameters.DEBUG_TENSOR_B.setBooleanProperty(false);
//        JNIDebugParameters.DEBUG_SOLVER_CONVERGENCE.setBooleanProperty(true);
    }

    @Override
    public SolverStat solveSystem() {
        int err = 0; /// jni_success
        if (ModelRegistry.constraintCounter == 0) {
            reporter.writeln("[solver/gpu] no constraints registered on model");
            return null;
        }
        long nextComputation = ModelRegistry.computationSnapshotId();
        reporter.writelnf("[solver/gpu]  ! snapshot id == %d", nextComputation);
        if (lastSnapshotId != nextComputation) {
            /* register model after structural changes */
            err = destroyComputation();
            if (err != JNI_SUCCESS) {
                reporter.writeln("[solver/gpu] destroy computation error !");
                return null;
            }

            boolean registered = registerModelOnGPU();
            if (!registered) {
                reporter.writeln("[solver/gpu] model registration error !");
                return null;
            }
            reporter.writeln("[solver/gpu] model registration OK !");

            err = initComputation();
            if (err != JNI_SUCCESS) {
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

        /* ------------------------------------------------------------
         * ------------------------- SOLVER ---------------------------
         * ------------------------------------------------------------ */
        try {
            err = solveSystemExecute();

        } catch (Exception e) {
            reporter.writelnf("[solver/gpu] solver execution failed %s!", e.getMessage());
            e.printStackTrace();
            return null;
        }

        if (err != JNI_SUCCESS) {
            reporter.writelnf("[solver/gpu] solver execution failed with error = %s!", lastError());
            return null;
        }

        final SolverStat solverStatistics = getSolverStatistics();

        reporter.writelnf("[solver/gpu] computation success  !");
        reporter.writelnf(" -- computation lastError    : %s", lastError());
        reporter.writelnf(" -- computation convergence  : %b", solverStatistics.convergence);

        boolean isResultValid = !Double.isNaN(solverStatistics.error);
        if (isResultValid) {
            fetchGPUComputedPositionsIntoModel();
        }
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
        updateConstrainState(size, constraintId, vecX, vecY);
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
        updateStateVector(stateVector);
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
        updateParametersValues(size, value, parameterId);
    }

    @Override
    public void destroyDriver() {

        int error = destroyComputation();
        if (error != JNI_SUCCESS) {
            reporter.writelnf("[gpu/solver] destroy computation error !");
        }

        error = destroyComputationContext();
        if (error != JNI_SUCCESS) {
            reporter.writelnf("[gpu/solver] destroy computation context error !");
        }

        error = closeDriver();
        if (error != JNI_SUCCESS) {
            reporter.writelnf("[gpu/solver] close driver error !");
        }
    }

    /**
     * Fetch all computed points according to state vector ordering by Point[ID] property.
     *
     * Data in moved from State Vector of gpu geometric solver.
     */
    private void fetchGPUComputedPositionsIntoModel() {
        double[] coordinateVector = fetchStateVector();
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
            int err = registerPointType(id, px, py);
            if (err != JNI_SUCCESS) {
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
            int err = registerGeometricType(primitiveId, geometricType, p1, p2, p3, a, b, c, d);
            if (err != JNI_SUCCESS) {
                reporter.writelnf("[solver/gpu] point registration failed : %s", geometric.toString());
                return false;
            }
        }

        // register parameter set
        for (Parameter parameter : ModelRegistry.dbParameter().values()) {
            int parameterId = parameter.getId();
            double value = parameter.getValue();
            int err = registerParameterType(parameterId, value);
            if (err != JNI_SUCCESS) {
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
            int err = registerConstraintType(constraintId, constraintTypeId, k, l, m, n, paramId, vecX, vecY);
            if (err != JNI_SUCCESS) {
                reporter.writelnf("[solver/gpu] constraint registration failed : %s", constraint.toString());
                return false;
            }
        }
        /// Model registered properly without conflicts !
        return true;
    }

    private int initDriver(int deviceId) {
        return JNISolverGate.initDriver(deviceId);
    }


    private String lastError() {
        return JNISolverGate.getLastError();
    }

    private int initComputationContext() {
        return JNISolverGate.initComputationContext();
    }

    private SolverStat getSolverStatistics() {
        return JNISolverGate.getSolverStatistics();
    }

    private int solveSystemExecute() {
        return JNISolverGate.solveSystem();
    }

    private int initComputation() {
        return JNISolverGate.initComputation();
    }

    private int destroyComputation() {
        return JNISolverGate.destroyComputation();
    }

    private double[] fetchStateVector() {
        return JNISolverGate.fetchStateVector();
    }

    private int registerPointType(int id, double px, double py) {
        return JNISolverGate.registerPointType(id, px, py);
    }

    private int registerConstraintType(int constraintId, int constraintTypeId, int k, int l, int m, int n, int paramId, double vecX, double vecY) {
        return JNISolverGate.registerConstraintType(constraintId, constraintTypeId, k, l, m, n, paramId, vecX, vecY);
    }

    private int registerGeometricType(int primitiveId, int geometricType, int p1, int p2, int p3, int a, int b, int c, int d) {
        return JNISolverGate.registerGeometricType(primitiveId, geometricType, p1, p2, p3, a, b, c, d);
    }

    private int registerParameterType(int parameterId, double value) {
        return JNISolverGate.registerParameterType(parameterId, value);
    }

    private void updateConstrainState(int size, int[] constraintId, double[] vecX, double[] vecY) {
        JNISolverGate.updateConstraintState(constraintId, vecX, vecY, size);
    }

    private void updateStateVector(double[] stateVector) {
        JNISolverGate.updateStateVector(stateVector);
    }

    private int closeDriver() {
        return JNISolverGate.closeDriver();
    }

    private int destroyComputationContext() {
        return JNISolverGate.destroyComputationContext();
    }

    private void updateParametersValues(int size, double[] value, int[] parameterId) {
        JNISolverGate.updateParametersValues(parameterId, value, size);
    }

}

