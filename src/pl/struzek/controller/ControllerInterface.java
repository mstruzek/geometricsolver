package pl.struzek.controller;

import pl.struzek.msketch.GeometricConstraintType;
import pl.struzek.msketch.Parameter;
import pl.struzek.msketch.Vector;

/**
 * Interfejs do oddzialywania na Controler(Model)
 * 
 * @author root
 *
 */
public interface ControllerInterface {

	/** Dodaj linie */
	public void addLine(Vector v1,Vector v2);
	
	/** Dodaj kolo */
	public void addCircle(Vector v1,Vector v2);
	
	/** Dodaj Luk */
	public void addArc(Vector v1,Vector v2);
	
	/** Dodaj punkt */
	public void addPoint(Vector v1);
	
	/** Rozwiaz system */
	public void solveSystem();
	
	/** Przelicz punkty kontrolne -wyzeruj sily */
	public void relaxForces();
	
	/** Dodaj wiez do modelu */
	void addConstraint(GeometricConstraintType constraintType, int K, int L,int M, int N, double d);
}
