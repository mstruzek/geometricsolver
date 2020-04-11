package com.mstruzek.controller;

import java.util.ArrayList;

import com.mstruzek.graphic.TView;
import com.mstruzek.msketch.GeometricConstraintType;
import com.mstruzek.msketch.GeometricPrymitive;
import com.mstruzek.msketch.Model;
import com.mstruzek.msketch.Vector;
import com.mstruzek.msketch.Model.MyTableModel;

public class Controller implements ControllerInterface{

	/** model szkicownika */
	Model sketchModel = null;
	
	public Controller() {
		sketchModel = new Model();

	}
	
	@Override
	public void addLine(Vector v1, Vector v2) {
		sketchModel.addLine(v1,v2);
	}

	@Override
	public void addCircle(Vector v1, Vector v2) {
		sketchModel.addCircle(v1,v2);
	}

	@Override
	public void addArc(Vector v1, Vector v2) {
		sketchModel.addArc(v1,v2);	
	}

	@Override
	public void addPoint(Vector v1) {
		sketchModel.addPoint(v1);	
	}

	@Override
	public void addConstraint(GeometricConstraintType constraintType, int K,int L, int M, int N, double d) {
		sketchModel.addConstraint(constraintType,K,L,M,N,d);
		
	}

	/** 
	 * Pobierz TableModel wiezow,parametrow, prymitywow
	 * @return
	 */
	public MyTableModel getConstraintTableModel(){
		return sketchModel.getConstraintTableModel();
		
	}
	public MyTableModel getParametersTableModel(){
		return sketchModel.getParametersTableModel();
		
	}
	public MyTableModel getPrimitivesTableModel(){
		return sketchModel.getPrimitivesTableModel();
		
	}
	public ArrayList<GeometricPrymitive> getPrimitivesContainer(){
		return sketchModel.getPrimitivesContainer();
	}
	@Override
	public void solveSystem() {
		sketchModel.solveSystem();
		
	}
	
	@Override
	public void relaxForces() {
		sketchModel.relaxForces();
		
	}
	
	public static void main(String[] args) {
	
		Controller controller =  new Controller();
		//TView view  =
		new TView("M-Sketcher",controller);
	}


	
}
