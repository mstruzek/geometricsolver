package com.mstruzek.controller;

import com.mstruzek.graphic.FrameView;
import com.mstruzek.msketch.*;
import com.mstruzek.msketch.solver.GeometricSolverType;

import java.io.File;
import java.util.Set;
import java.util.TreeMap;


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

        try (GCModelWriter modelWriter = new GCModelWriter(selectedFile)) {

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
    }

    public void readModelFrom(File selectedFile) {

        ModelRegistry.removeObjectsFromModel();

        try (GCModelReader modelReader = new GCModelReader(selectedFile)) {

            modelReader.readModel();

            updateModelConsistency();

        } catch (Exception e) {
            Reporter.notify("[error] read model from file : " + selectedFile, e);
            throw new Error(e);
        }
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
    public void relaxControlPoints(double coefficient) {
        Model.relaxControlPoints(coefficient);
    }

    public static void main(String[] args) {

        Controller controller = new Controller();
        //TView view  =
        new FrameView(controller);
    }
}
