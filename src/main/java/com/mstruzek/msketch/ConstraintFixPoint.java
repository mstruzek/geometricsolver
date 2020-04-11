package com.mstruzek.msketch;

import java.util.TreeMap;

import Jama.Matrix;

import com.mstruzek.msketch.matrix.MatrixDouble;

/**
 * Klasa reprezentuje wiez typu FIX : "PointK - VectorK0 =0";
 * @author root
 *
 */
public class ConstraintFixPoint extends Constraint{

	/** Point K-id */
	int k_id;
	/** Vector L*/
	Vector k0_vec= null;

	/**
	 * Konstruktor wiezu
	 * @param K - Point zafiksowany bedzie w aktualnym miejscu
	 */
	public ConstraintFixPoint(Point K) {
		super(GeometricConstraintType.FixPoint);
		//pobierz id
		k_id = K.id;
		k0_vec=new Vector((Vector)K);
		dbConstraint.put(constraintId,this);
	}
	
	/**
	 * Konstruktor wiezu
	 * @param K - zrzutowany Point na Vector
	 * @param L - Vector w ktorym bedzie zafiksowany K
	 */
	public ConstraintFixPoint(Point K,Vector L) {
		super(GeometricConstraintType.FixPoint);
		//pobierz id
		k_id = ((Point)K).id;
		k0_vec=L;
		dbConstraint.put(constraintId,this);
	}	
	
	/**
	 * Funkcja ustawia nowy wektor zafiksowania
	 * @param vc
	 */
	public void setFixVector(Vector vc){
		k0_vec = vc;
	}
	
	public String toString(){
		MatrixDouble out = getValue(Point.dbPoint, Parameter.dbParameter);
		double norm = Matrix.constructWithCopy(out.getArray()).norm1();
		
		return "Constraint-FixPoint" + constraintId + "*s" + size() + " = " + norm + " { K =" + Point.dbPoint.get(k_id) + "  , K0 = " + k0_vec + " } \n";
	}

	@Override
	public MatrixDouble getJacobian(TreeMap<Integer, Point> dbPoints,TreeMap<Integer, Parameter> dbParameter) {
		//macierz 2 wierszowa
		MatrixDouble out = MatrixDouble.fill(2,dbPoints.size()*2,0.0);
		//zerujemy cala macierz + wstawiamy na odpowiednie miejsce Jacobian wiezu
		int j=0;
		for(Integer i:dbPoints.keySet()){
			if(k_id==dbPoints.get(i).id){
				//macierz jednostkowa
				out.m[0][j*2]=1.0;out.m[0][j*2+1] =0.0;
				out.m[1][j*2]=0.0;out.m[1][j*2+1] =1.0;				
			}			
			j++;
		}	
		return out;
	}

	@Override
	public boolean isJacobianConstant() {
		return true;
	}

	@Override
	public MatrixDouble getValue(TreeMap<Integer, Point> dbPoints,TreeMap<Integer, Parameter> dbParameter) {
		return new MatrixDouble(((Vector)dbPoints.get(k_id)).sub(k0_vec),true);
	}

	@Override
	public MatrixDouble getHessian(TreeMap<Integer, Point> dbPoints,TreeMap<Integer, Parameter> dbParameter) {
		return null;
	}

	@Override
	public boolean isHessianConstant() {
		return false;
	}

	
	@Override
	public int getK() {
		return k_id;
	}
	@Override
	public int getL() {
		return -1;
	}
	@Override
	public int getM() {
		return -1;
	}
	@Override
	public int getN() {
		return -1;
	}
	@Override
	public int getParametr() {
		return -1;
	}
	
	public static void main(String[] args) {
		
		Point pn =new Point(new Vector(3.0,4.0));
		Point pn3 =new Point(new Vector(3.0,4.0));
		Point pn2 = new Point(new Vector(4.0,5.0));
		
		ConstraintConect2Points conectPoint = new ConstraintConect2Points(pn,pn2);
		
		
		ConstraintFixPoint fixPoint2 = new ConstraintFixPoint(pn3);
		System.out.println(Constraint.dbConstraint );
		//jakobian z wiezow
		//double[][] tab = fixPoint2.getJacobian2D(Point.dbPoint, Parameter.dbParameter);
		//MatrixDouble tab2 = conectPoint.getJacobian(Point.dbPoint, Parameter.dbParameter);
		
		//tak sie zabieramy za wielkosc wektora
		System.out.println(fixPoint2.getJacobian(Point.dbPoint, Parameter.dbParameter));
		System.out.println(conectPoint.getJacobian(Point.dbPoint, Parameter.dbParameter));

	}

	@Override
	public double getNorm(TreeMap<Integer, Point> dbPoints,TreeMap<Integer, Parameter> dbParameter) {
		
		MatrixDouble md = getValue(dbPoints, dbParameter);
		double val = Math.sqrt(md.m[0][0]*md.m[0][0]+md.m[1][0]*md.m[1][0]);
		return val;
	}
}
