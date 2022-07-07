package com.mstruzek.controller;

import java.io.File;
import java.util.ArrayList;

import com.mstruzek.graphic.TView;
import com.mstruzek.msketch.*;
import com.mstruzek.msketch.MyTableModel;

public class Controller implements ControllerInterface{

    private static MyTableModel primitivesTableModel=null;
    private static MyTableModel constraintTableModel=null;
    private static MyTableModel parametersTableModel=null;

    /**
     * model szkicownika
     */

    public Controller(){
    }

    @Override
    public void addLine(Vector v1,Vector v2){
        Model.addLine(v1,v2);
    }

    @Override
    public void addCircle(Vector v1,Vector v2){
        Model.addCircle(v1,v2);
    }

    @Override
    public void addArc(Vector v1,Vector v2){
        Model.addArc(v1,v2);
    }

    @Override
    public void addPoint(Vector v1){
        Model.addPoint(v1);
    }

    @Override
    public void addConstraint(GeometricConstraintType constraintType,int K,int L,int M,int N,double d){
        Model.addConstraint(constraintType,K,L,M,N,d);

    }

    public void persistModel(File filePath){
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

        try(GCModelWriter modelWriter=new GCModelWriter(filePath)){
            modelWriter.writeHeader();
            modelWriter.writePoints();
            modelWriter.writeGeometricPrimitives();
            modelWriter.writeConstraints();
            modelWriter.writeParameters();
            modelWriter.writeClose();
        }catch(Exception e){
            throw new RuntimeException(e);
        }
    }

    /**
     * Zwraca widok na model wiezow dodanych przez uzytkownika
     */
    public MyTableModel getConstraintTableModel(){
        if(constraintTableModel==null){
            constraintTableModel= createConstraintTableModel();
        }
        return constraintTableModel;
    }

    public MyTableModel getParametersTableModel(){
        if(parametersTableModel==null){
            parametersTableModel= createParametersTableModel();
        }
        return parametersTableModel;
    }

    public MyTableModel getPrimitivesTableModel(){
        if(primitivesTableModel==null){
            primitivesTableModel= createPrimitivesTableModel();
        }
        return primitivesTableModel;
    }

    /**
     * Zwraca widok na model elementow geometrycznych
     */
    private MyTableModel createPrimitivesTableModel() {
        return new PrimitivesTableModel();
    }

    /**
     * Zwraca widok na parametry
     */
    public MyTableModel createParametersTableModel() {
        return new ParametersTableModel();
    }

    public MyTableModel createConstraintTableModel() {
        return new ConstraintsTableModel();
    }

    public ArrayList<GeometricPrimitive> getPrimitivesContainer(){
        return Model.primitivesContainer();
    }

    @Override
    public void solveSystem(){
        Model.solveSystem();
    }

    @Override
    public void relaxForces(){
        Model.relaxForces();
    }

    @Override
    public void fluctuatePoints(double coefficient){
        Model.fluctuatePoints(coefficient);
    }

    public static void main(String[] args){

        Controller controller=new Controller();
        //TView view  =
        new TView("M-Sketcher",controller);
    }

    public void unPersistModel(File selectedFile){

    }

}
