package com.mstruzek.msketch;

import java.util.TreeMap;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

/**
 * Klasa reprezentujaca wiez rownej dlugosci pomiedzy
 * dwoma wektorami (4 punkty)
 * @author root
 *
 */
public class ConstraintLinesSameLength2 extends Constraint {

	/** Punkty kontrolne */
	/** Point K-id */
	int k_id;
	/** Point L-id */
	int l_id;
	/** Point M-id */
	int m_id;
	/** Point N-id */
	int n_id;
	
	/**
	 * Konstruktor pomiedzy 4 punktami lub
	 * 2 punktami i FixLine(czyli 2 wektory)
	 * rownanie tego wiezu to [(K-L)'*(K-L)] - [(M-N)'*(M-N)] = 0 
	 * Gorsze wlasciwosci numeryczne 
	 * @param K
	 * @param L
	 * @param M
	 * @param N
	 */
	public ConstraintLinesSameLength2(int constId, Point K, Point L ,Point M,Point N){
		super(constId, GeometricConstraintType.LinesSameLength, true);
		k_id = K.id;
		l_id = L.id;
		m_id = M.id;
		n_id = N.id;
	}

	public String toString(){
		MatrixDouble out = getValue(Point.dbPoint, Parameter.dbParameter);
		double norm = Matrix.constructWithCopy(out.getArray()).norm1();
		return "Constraint-LinesSameLength2" + constraintId + "*s" + size() + " = " + norm  + " { K =" + Point.dbPoint.get(k_id) + "  ,L =" + Point.dbPoint.get(l_id) + " ,M =" + Point.dbPoint.get(m_id) + ",N =" + Point.dbPoint.get(n_id) + "} \n";
	
	}
	
	@Override
	public MatrixDouble getJacobian(TreeMap<Integer, Point> dbPoints,TreeMap<Integer, Parameter> dbParameter) {
		//macierz 2 wierszowa
		MatrixDouble out = MatrixDouble.fill(1,dbPoints.size()*2,0.0);
		//zerujemy cala macierz + wstawiamy na odpowiednie miejsce Jacobian wiezu
		int j=0;
		Vector vLK = ((Vector)dbPoints.get(k_id)).sub((Vector)dbPoints.get(l_id)).dot(2.0);
		Vector vNM = ((Vector)dbPoints.get(m_id)).sub((Vector)dbPoints.get(n_id)).dot(2.0);
		for(Integer i:dbPoints.keySet()){
			
			//a tu wstawiamy macierz dla tego wiezu
			if(k_id==dbPoints.get(i).id){
				out.m[0][j*2]= vLK.x ;out.m[0][j*2+1] = vLK.y;
			}	
			if(l_id==dbPoints.get(i).id){
				out.m[0][j*2]= -vLK.x ;out.m[0][j*2+1] = -vLK.y;				
			}
			//a tu wstawiamy macierz dla tego wiezu
			if(m_id==dbPoints.get(i).id){
				out.m[0][j*2]= -vNM.x ;out.m[0][j*2+1] = -vNM.y;
			}	
			if(n_id==dbPoints.get(i).id){
				out.m[0][j*2]= vNM.x ;out.m[0][j*2+1] = vNM.y;			
			}
			j++;
		}			
	
		return out;
	}

	@Override
	public boolean isJacobianConstant() {
		return false;
		
	}
	@Override
	public MatrixDouble getValue(TreeMap<Integer, Point> dbPoints,TreeMap<Integer, Parameter> dbParameter) {

		Double vLK = ((Vector)dbPoints.get(k_id)).sub((Vector)dbPoints.get(l_id)).length();
		Double vNM = ((Vector)dbPoints.get(m_id)).sub((Vector)dbPoints.get(n_id)).length();		
		MatrixDouble mt = new MatrixDouble(1,1);
		mt.m[0][0] = vLK*vLK-vNM*vLK;
		return mt;
	}

	@Override
	public MatrixDouble getHessian(TreeMap<Integer, Point> dbPoints,TreeMap<Integer, Parameter> dbParameter) {

		//macierz NxN
		MatrixDouble out = MatrixDouble.fill(dbPoints.size()*2,dbPoints.size()*2,0.0);
		MatrixDouble I = MatrixDouble.identity(2).dot(2.0);
		MatrixDouble mI = MatrixDouble.identity(2).dot(-2.0);
		
		for(Integer i:dbPoints.keySet()){ //wiersz
			for(Integer j:dbPoints.keySet()){ //kolumna
				//wstawiamy I,-I w odpowiednie miejsca
				//k,k
				if(k_id==dbPoints.get(i).id && k_id==dbPoints.get(j).id ){
					out.addSubMatrix(2*i, 2*j, I);
					
				}
				//k,l
				if(k_id==dbPoints.get(i).id && l_id==dbPoints.get(j).id ){
					out.addSubMatrix(2*i, 2*j, mI);
				}
				//l,k
				if(l_id==dbPoints.get(i).id && k_id==dbPoints.get(j).id ){
					out.addSubMatrix(2*i, 2*j, mI);
				}
				//l,l
				if(l_id==dbPoints.get(i).id && l_id==dbPoints.get(j).id ){
					out.addSubMatrix(2*i, 2*j, I);
				}
				//m,m
				if(m_id==dbPoints.get(i).id && m_id==dbPoints.get(j).id ){
					out.addSubMatrix(2*i, 2*j, mI);
				}
				//m,n
				if(m_id==dbPoints.get(i).id && n_id==dbPoints.get(j).id ){
					out.addSubMatrix(2*i, 2*j, I);
				}
				//n,m
				if(n_id==dbPoints.get(i).id && m_id==dbPoints.get(j).id ){
					out.addSubMatrix(2*i, 2*j, I);
				}
				//n,n
				if(n_id==dbPoints.get(i).id && n_id==dbPoints.get(j).id ){
					out.addSubMatrix(2*i, 2*j, mI);
				}
			}	
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


	}
	@Override
	public double getNorm(TreeMap<Integer, Point> dbPoints,TreeMap<Integer, Parameter> dbParameter) {

		return getValue(dbPoints,dbParameter).m[0][0];
	}

}
