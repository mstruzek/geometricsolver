package com.mstruzek.msketch;

import java.util.TreeMap;

import Jama.Matrix;

import com.mstruzek.msketch.matrix.MatrixDouble;

/**
 * Klasa reprezentujaca wiez rownoleglo�ci pomiedzy 
 * liniami(vectorami)
 * @author root
 *
 */
public class ConstraintLinesParallelism extends Constraint{
	

	/** Punkty kontrolne */
	/** Point K-id */
	int k_id;
	/** Point L-id */
	int l_id;
	/** Point M-id */
	int m_id;
	/** Vector M - gdy wiez pomiedzy fixline*/
	Vector m=null;
	/** Point N-id */
	int n_id;
	/** Vector N - gdy wiez pomiedzy fixline*/
	Vector n=null;
	
	/** macierz rotacji o 90 stopni */
	static MatrixDouble R = MatrixDouble.getRotation2x2(90+180);
	static MatrixDouble mR = MatrixDouble.getRotation2x2(90); //mR=-R
	
	/**
     * Konstruktor pomiedzy 4 punktami lub
     * 2 punktami i FixLine(czyli 2 wektory)
     * rownanie tego wiezu to (L-K)x(N-M) = 0
     * iloczyn wektorowy
     * FIXME - zastanowic sie czy nie zrobic abs((L-K)x(N-M)) =0 moze bedzie szybciej zbiegal
     *
     * @param constId
     * @param K
     * @param L
     * @param M
     * @param N
     */
	public ConstraintLinesParallelism(int constId,Point K,Point L ,Vector M,Vector N){
		super(constId, GeometricConstraintType.LinesParallelism);
		
		k_id = K.id;
		l_id = L.id;
		if((M instanceof Point) && ( N instanceof Point)){
			m_id =((Point)M).id;
			n_id = ((Point)N).id;
		}else{
			m=M;
			n= N;			
		}
	}
	public String toString(){
		MatrixDouble out = getValue(Point.dbPoint, Parameter.dbParameter);
		double norm = Matrix.constructWithCopy(out.getArray()).norm1();
		if(m==null && n==null) return "Constraint-LinesParallelism" + constraintId + "*s" + size() + " = " + norm + " { K =" + Point.dbPoint.get(k_id) + "  ,L =" + Point.dbPoint.get(l_id) + " ,M =" + Point.dbPoint.get(m_id) + ",N =" + Point.dbPoint.get(n_id) + "} \n";
		else{
			return "Constraint-LinesParallelism" + constraintId + "*s" + size() + " = " + norm + " { K =" + Point.dbPoint.get(k_id) + "  ,L =" + Point.dbPoint.get(l_id) + " ,vecM =" + m + ",vecN =" + n + "} \n";
		}
		
	}
	
	@Override
	public MatrixDouble getValue(TreeMap<Integer, Point> dbPoints,TreeMap<Integer, Parameter> dbParameter) {
		Vector out = new Vector(dbPoints.get(l_id));
		out = out.sub((Vector)dbPoints.get(k_id));
		//out =out.unit();
		
		MatrixDouble mt = new MatrixDouble(1,1);
		
		if((m==null) && ( n ==null)){
			mt.m[0][0] = out.cross(((Vector)dbPoints.get(n_id)).sub((Vector)dbPoints.get(m_id)));
		}else{
			mt.m[0][0] = out.cross(n.sub(m));
		}
		return mt;
	}
	@Override
	public MatrixDouble getJacobian(TreeMap<Integer, Point> dbPoints,TreeMap<Integer, Parameter> dbParameter) {
		//macierz 2 wierszowa
		MatrixDouble out = MatrixDouble.fill(1,dbPoints.size()*2,0.0);
		//zerujemy cala macierz + wstawiamy na odpowiednie miejsce Jacobian wiezu
		int j=0;
		if((m==null) && ( n ==null)){
			for(Integer i:dbPoints.keySet()){
				
				//a tu wstawiamy macierz dla tego wiezu
				if(k_id==dbPoints.get(i).id){
					Vector v1 = ((Vector)dbPoints.get(n_id)).sub((Vector)dbPoints.get(m_id));
					out.m[0][j*2]+= -v1.y ;out.m[0][j*2+1] = v1.x;
				}	
				if(l_id==dbPoints.get(i).id){
					Vector v1 = ((Vector)dbPoints.get(n_id)).sub((Vector)dbPoints.get(m_id));
					out.m[0][j*2]+= v1.y ;out.m[0][j*2+1] = -v1.x;				
				}
				//a tu wstawiamy macierz dla tego wiezu
				if(m_id==dbPoints.get(i).id){
					Vector v1 = ((Vector)dbPoints.get(l_id)).sub((Vector)dbPoints.get(k_id));
					out.m[0][j*2]+= v1.y ;out.m[0][j*2+1] = -v1.x;
				}	
				if(n_id==dbPoints.get(i).id){
					Vector v1 = ((Vector)dbPoints.get(l_id)).sub((Vector)dbPoints.get(k_id));
					out.m[0][j*2]+= -v1.y ;out.m[0][j*2+1] = v1.x;			
				}
				j++;
			}			
		}else{
			Vector v1 = n.sub(m);
			for(Integer i:dbPoints.keySet()){
				
				//a tu wstawiamy macierz dla tego wiezu
				if(k_id==dbPoints.get(i).id){				
					out.m[0][j*2]+= -v1.y ;out.m[0][j*2+1] = v1.x;
				}	
				if(l_id==dbPoints.get(i).id){
					out.m[0][j*2]+= v1.y ;out.m[0][j*2+1] = -v1.x;				
				}
				//reszta dla m,n =0
				j++;
			}				
		}
	
		return out;
	}

	@Override
	public boolean isJacobianConstant() {
		if((m==null) && ( n ==null)){
			return false;
		}else{
			//jezeli m,n vectory to constatnt i wtedy Hessian =0
			return true;
		}
		
	}
	@Override
	public MatrixDouble getHessian(TreeMap<Integer, Point> dbPoints,TreeMap<Integer, Parameter> dbParameter) {

		//macierz NxN
		MatrixDouble out = MatrixDouble.fill(dbPoints.size()*2,dbPoints.size()*2,0.0);

		if((m==null) && ( n ==null)){
			//same punkty
			for(Integer i:dbPoints.keySet()){ //wiersz
				for(Integer j:dbPoints.keySet()){ //kolumna
					//wstawiamy I,-I w odpowiednie miejsca
					//k,m
					if(k_id==dbPoints.get(i).id && m_id==dbPoints.get(j).id ){
						out.addSubMatrix(2*i, 2*j, R);
						
					}
					//k,n
					if(k_id==dbPoints.get(i).id && n_id==dbPoints.get(j).id ){
						out.addSubMatrix(2*i, 2*j, mR);
					}
					//l,m
					if(l_id==dbPoints.get(i).id && m_id==dbPoints.get(j).id ){
						out.addSubMatrix(2*i, 2*j, mR);
					}
					//l,n
					if(l_id==dbPoints.get(i).id && n_id==dbPoints.get(j).id ){
						out.addSubMatrix(2*i, 2*j, R);
					}
					//m,k
					if(m_id==dbPoints.get(i).id && k_id==dbPoints.get(j).id ){
						out.addSubMatrix(2*i, 2*j, mR);
					}
					//m,l
					if(m_id==dbPoints.get(i).id && l_id==dbPoints.get(j).id ){
						out.addSubMatrix(2*i, 2*j, R);
					}
					//n,k
					if(n_id==dbPoints.get(i).id && k_id==dbPoints.get(j).id ){
						out.addSubMatrix(2*i, 2*j, R);
					}
					//n,l
					if(n_id==dbPoints.get(i).id && l_id==dbPoints.get(j).id ){
						out.addSubMatrix(2*i, 2*j, mR);
					}
				}
				
			}
			
		}else{
			//m,n - vectory
			// HESSIAN ZERO
			
		}
		return out;
	}
	@Override
	public boolean isHessianConstant() {
		return true;
	}	
	@Override
	public int getK() {
		return k_id;
	}
	@Override
	public int getL() {
		return l_id;
	}
	@Override
	public int getM() {
		return m_id;
	}
	@Override
	public int getN() {
		return n_id;
	}
	@Override
	public int getParametr() {
		return -1;
	}
	/**
	 * @param args
	 */
	public static void main(String[] args) {

		Point pn1 = new Point(0.0,0.0);
		Point pn2 = new Point(10.0,0.0);
		Point pn3 = new Point(1.,1.0);
		Point pn4 = new Point(10.0,1.1);

		ConstraintLinesParallelism cn = new ConstraintLinesParallelism(Constraint.nextId(),pn1,pn2,pn3,pn4);
		System.out.println(Constraint.dbConstraint );
		System.out.println(cn.getJacobian(Point.dbPoint, Parameter.dbParameter));
		System.out.println(cn.getValue(Point.dbPoint, Parameter.dbParameter));
		System.out.println(cn.getNorm(Point.dbPoint, Parameter.dbParameter));

	}
	@Override
	public double getNorm(TreeMap<Integer, Point> dbPoints,TreeMap<Integer, Parameter> dbParameter) {
		
		double val = getValue( dbPoints,dbParameter).m[0][0];
		
		Vector out = new Vector(((Vector)dbPoints.get(k_id)).sub((Vector)dbPoints.get(l_id)));
		val=val/out.length();
		
		if((m==null) && ( n ==null)){
			val=val/((Vector)dbPoints.get(m_id)).sub((Vector)dbPoints.get(n_id)).length();
		}else{
			val=val/(m.sub(n)).length();
		}
		
		return val;
	}
}
