package com.mstruzek.controller;

import com.mstruzek.graphic.FrameView;
import com.mstruzek.msketch.*;

import java.io.File;
import java.util.Set;
import java.util.TreeMap;

public class Controller implements ControllerInterface {

    /**
     * model szkicownika
     */

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
    public void addConstraint(GeometricConstraintType constraintType, int K, int L, int M, int N, double d) {
        Model.addConstraint(constraintType, K, L, M, N, d);

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
            modelWriter.writeGeometricPrimitives();
            modelWriter.writeParameters();
            modelWriter.writeConstraints();
            modelWriter.writeClose();

        } catch (Exception e) {
            Reporter.notify("[error] write model into file : " + selectedFile, e);
            throw new Error(e);
        }
    }

    public void readModelFrom(File selectedFile) {

        clearDatabasesModel();

        try (GCModelReader modelReader = new GCModelReader(selectedFile)) {

            modelReader.readModel();

            updateModelConsistency();

        } catch (Exception e) {
            Reporter.notify("[error] read model from file : " + selectedFile, e);
            throw new Error(e);
        }
    }

    private void clearDatabasesModel() {
        Constraint.dbConstraint.clear();
        Constraint.constraintCounter = 0;

        Parameter.dbParameter.clear();
        Parameter.parameterCounter = 0;

        GeometricPrimitive.dbPrimitives.clear();
        GeometricPrimitive.primitiveCounter = 0;

        Point.dbPoint.clear();
        Point.pointCounter = 0;

    }

    private void updateModelConsistency() {
        // Update State and related variables !
        Point.pointCounter = firstAvailableKey(Point.dbPoint);

        GeometricPrimitive.primitiveCounter = firstAvailableKey(GeometricPrimitive.dbPrimitives);

        Parameter.parameterCounter = firstAvailableKey(Parameter.dbParameter);

        Set<Integer> skipConstrainIds = Constraint.dbConstraint.keySet();
        for(GeometricPrimitive geometricPrimitive : GeometricPrimitive.dbPrimitives.values()) {
            geometricPrimitive.setAssociateConstraints(skipConstrainIds);
        }

        Constraint.constraintCounter = firstAvailableKey(Constraint.dbConstraint);
    }

    private <ModelEntity> int firstAvailableKey(TreeMap<Integer,ModelEntity> treeMap) {
        return treeMap.size() == 0 ? 0 : treeMap.lastKey() + 1;
    }


    @Override
    public void solveSystem() {
        Model.solveSystem();
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
        new FrameView("M-Sketcher", controller);
    }
}
