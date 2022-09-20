package com.mstruzek.controller;

import com.mstruzek.graphic.FrameView;
import com.mstruzek.msketch.*;
import com.mstruzek.msketch.solver.GeometricSolverType;
import com.mstruzek.msketch.solver.SolverStat;

import java.io.File;
import java.util.Set;
import java.util.TreeMap;

import static com.mstruzek.controller.EventType.REBUILD_TABLES;


public class Controller implements ControllerInterface {

    private GeometricSolverType solverType = GeometricSolverType.CPU_SOLVER;

    public Controller() {
    }

    @Override
    public void addLine(Vector v1, Vector v2) {
        Model.addLine(v1, v2);
    }

    @Override
    public void addCircle(Vector v1, Vector v2) {
        Model.addCircle(v1, v2);
    }

    @Override
    public void addArc(Vector v1, Vector v2, Vector v3) {
        Model.addArc(v1, v2, v3);
    }

    @Override
    public void addPoint(Vector v1) {
        Model.addPoint(v1);
    }

    @Override
    public void addConstraint(ConstraintType constraintType, int K, int L, int M, int N, double d) {
        try {
            Model.addConstraint(constraintType, K, L, M, N, d);
        } catch (Exception e) {
            e.printStackTrace();
            Events.send(EventType.CONTROLLER_ERROR, new Object[]{e.getMessage()});
        }
    }

    @Override
    public void setSolverType(GeometricSolverType solverType) {
        this.solverType = solverType;
    }

    @Override
    public void shutdown() {
        Model.shutdown();
    }

    public void writeModelInto(File selectedFile) {
        /*
         * Persist state :
         *
         *  Geometric Primitives,
         *
         *  Constraints,
         *
         *  Parameters
         *
         *  into GCM file (Geometric Constraint Model - file suffixed with *.gcm extension)
         */

        String absolutePath = selectedFile.getAbsolutePath();
        String fileExtension = absolutePath.substring(absolutePath.lastIndexOf("."));

        switch (fileExtension) {
            case ".cpp":
                try (var modelWriter = new CppModelWriter(selectedFile)) {
                    modelWriter.writeHeader();
                    modelWriter.writePoints();
                    modelWriter.writeGeometricObjects();
                    modelWriter.writeParameters();
                    modelWriter.writeConstraints();
                    modelWriter.writeClose();
                } catch (Exception e) {
                    Reporter.notify("[error] write model into file : " + selectedFile, e);
                    throw new Error(e);
                }
                break;

            case ".gcm":
                try (var modelWriter = new GCModelWriter(selectedFile)) {
                    modelWriter.writeHeader();
                    modelWriter.writePoints();
                    modelWriter.writeGeometricObjects();
                    modelWriter.writeParameters();
                    modelWriter.writeConstraints();
                    modelWriter.writeClose();
                } catch (Exception e) {
                    Reporter.notify("[error] write model into file : " + selectedFile, e);
                    throw new Error(e);
                }
                break;

        default:
                throw new Error("[error] no handler function for file output extension");
        }
    }

    public void readModelFrom(File selectedFile) {

        ModelRegistry.removeObjectsFromModel();

        try (GCModelReader modelReader = new GCModelReader(selectedFile)) {

            Events.DISABLE = true;
            try {
                modelReader.readModel();
            } finally {
                Events.DISABLE = false;
            }
            updateModelConsistency();

            Events.send(REBUILD_TABLES, new Object[]{});

        } catch (Exception e) {
            Reporter.notify("[error] read model from file : " + selectedFile, e);
            throw new Error(e);
        }

        final SolverStat stats = ModelRegistry.modelDefaultStats();
        Events.send(EventType.SOLVER_STAT_CHANGE, new Object[]{stats});
    }

    private void updateModelConsistency() {
        // Update State and related variables !
        ModelRegistry.pointCounter = firstAvailableKey(ModelRegistry.dbPoint());
        ModelRegistry.primitiveCounter = firstAvailableKey(ModelRegistry.dbPrimitives());
        ModelRegistry.parameterCounter = firstAvailableKey(ModelRegistry.dbParameter());

        Set<Integer> skipConstrainIds = ModelRegistry.dbConstraint.keySet();
        for (GeometricObject geometricObject : ModelRegistry.dbPrimitives.values()) {

            geometricObject.setAssociateConstraints(skipConstrainIds);

            for (Constraint constraint : geometricObject.associatedConstraints()) {
                ModelRegistry.registerConstraint(constraint.getConstraintId(), constraint);
            }
        }

        ModelRegistry.constraintCounter = firstAvailableKey(ModelRegistry.dbConstraint);
    }

    private static <ModelEntity> int firstAvailableKey(TreeMap<Integer, ModelEntity> treeMap) {
        return treeMap.size() == 0 ? 0 : treeMap.lastKey() + 1;
    }

    @Override
    public void solveSystem() {
        try {
            /*
             * *************************************
             *     Solve Linear Equation System
             * *************************************
             */
            Model.solveSystem(solverType);

            Events.send(EventType.REFRESH_N_REPAINT, new Object[]{});

        } catch (Throwable e) {
            e.printStackTrace();
            Events.send(EventType.CONTROLLER_ERROR, new Object[]{e.getMessage()});
        }
    }

    @Override
    public void evaluateGuidePoints() {
        Model.evaluateGuidePoints();
    }

    @Override
    public void relaxControlPoints() {
        Model.relaxControlPoints();
    }

    public static void main(String[] args) {

        Controller controller = new Controller();
        //TView view  =
        new FrameView(controller);
    }
}
