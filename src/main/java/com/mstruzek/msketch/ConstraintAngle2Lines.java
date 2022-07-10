package com.mstruzek.msketch;

import Jama.Matrix;

import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Parameter.dbParameter;
import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Wiez odpowiedzialny za kat pomiedzy wektorami
 * 
 * @author root
 *
 */
public class ConstraintAngle2Lines extends Constraint {

	/** Punkty kontrolne */
	/** Point K-id */
	int k_id;
	/** Point L-id */
	int l_id;
	/** Point M-id */
	int m_id;
	/** Point N-id */
	int n_id;
	/** Numer parametru przechowujacy kat w radianach */
	int param_id;

	
	/**
	 * Konstruktor pomiedzy 4 punktami i paramtetrem
	 * rownanie tego wiezu to (L-K)'*(N-M)-cos(param)*|L-K|*|N-M| = 0 
	 * @param K punkt prostej
	 * @param L punkt prostej
	 * @param M punkt prostej
	 * @param N punkt prostej
	 */
	public ConstraintAngle2Lines(Integer constId, Point K, Point L ,Point M,Point N,Parameter param){
		super(constId,GeometricConstraintType.Angle2Lines, true);
		k_id = K.id;
		l_id = L.id;
		m_id = M.id;
		n_id = N.id;
		param_id =param.getId();
	}

	public String toString(){
		MatrixDouble out = getValue();
		double norm = Matrix.constructWithCopy(out.getArray()).norm1();
		return "Constraint-Angle2Lines" + constraintId + "*s" + size() + " = " + norm  + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) + ", Parametr-" + dbParameter.get(param_id).getId() + " = "+ dbParameter.get(param_id).getValue() + "} \n";
	}
	
	@Override
	public MatrixDouble getValue() {
	
		Vector vLK = ((Vector)dbPoint.get(l_id)).sub((Vector)dbPoint.get(k_id));
		Vector vNM = ((Vector)dbPoint.get(n_id)).sub((Vector)dbPoint.get(m_id));
		
		MatrixDouble mt = new MatrixDouble(1,1);
		
		mt.m[0][0] = vLK.dot(vNM)-vLK.length()*vNM.length()*Math.cos(dbParameter.get(param_id).getRadians());
	
		return mt;
	}
	@Override
	public MatrixDouble getJacobian() {
		//macierz 2 wierszowa
		MatrixDouble out = MatrixDouble.fill(1,dbPoint.size()*2,0.0);
		//zerujemy cala macierz + wstawiamy na odpowiednie miejsce Jacobian wiezu
		int j=0;
		Vector vLK = ((Vector)dbPoint.get(l_id)).sub((Vector)dbPoint.get(k_id));
		Vector vNM = ((Vector)dbPoint.get(n_id)).sub((Vector)dbPoint.get(m_id));
		Vector uLKdNM = vLK.unit().dot(vNM.length()).dot(Math.cos(dbParameter.get(param_id).getRadians()));
		Vector uNMdLK = vNM.unit().dot(vLK.length()).dot(Math.cos(dbParameter.get(param_id).getRadians()));

		for(Integer i:dbPoint.keySet()){
			//a tu wstawiamy macierz dla tego wiezu
			if(k_id== dbPoint.get(i).id){
				out.m[0][j*2]= -vNM.x + uLKdNM.x;
				out.m[0][j*2+1] = -vNM.y + uLKdNM.y;
			}	
			if(l_id== dbPoint.get(i).id){
				out.m[0][j*2]= vNM.x - uLKdNM.x;
				out.m[0][j*2+1] = vNM.y - uLKdNM.y;		
			}
			//a tu wstawiamy macierz dla tego wiezu
			if(m_id== dbPoint.get(i).id){
				out.m[0][j*2]= -vLK.x + uNMdLK.x;
				out.m[0][j*2+1] = -vLK.y + uNMdLK.y;
			}	
			if(n_id== dbPoint.get(i).id){
				out.m[0][j*2]= vLK.x - uNMdLK.x;
				out.m[0][j*2+1] = vLK.y - uNMdLK.y;
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
	public MatrixDouble getHessian(double alfa) {
		//macierz NxN
		MatrixDouble out = MatrixDouble.fill(dbPoint.size()*2,dbPoint.size()*2,0.0);

		Vector vLK = ((Vector)dbPoint.get(l_id)).sub((Vector)dbPoint.get(k_id)).unit();
		Vector vNM = ((Vector)dbPoint.get(n_id)).sub((Vector)dbPoint.get(m_id)).unit();
		
		double g = vLK.dot(vNM)*Math.cos(dbParameter.get(param_id).getRadians());
		//same punkty
		int i = 0;
		for(Integer pI:dbPoint.keySet()){ //wiersz
			int j = 0;
			for(Integer pJ:dbPoint.keySet()){ //kolumna
				//k,k
				if(k_id== dbPoint.get(pI).id && k_id== dbPoint.get(pJ).id ){
					// 0
				}
				//k,l
				if(k_id== dbPoint.get(pI).id && l_id== dbPoint.get(pJ).id ){
					//0
				}
				//k,m
				if(k_id== dbPoint.get(pI).id && m_id== dbPoint.get(pJ).id ){
					out.setSubMatrix(2*i, 2*j,MatrixDouble.diagonal(2, 1-g));
				}
				//k,n
				if(k_id== dbPoint.get(pI).id && n_id== dbPoint.get(pJ).id ){
					out.setSubMatrix(2*i, 2*j,MatrixDouble.diagonal(2, g-1));
				}
				//l,k
				if(l_id== dbPoint.get(pI).id && k_id== dbPoint.get(pJ).id ){
					//0
				}
				//l,l
				if(l_id== dbPoint.get(pI).id && l_id== dbPoint.get(pJ).id ){
					// 0
				}
				//l,m
				if(l_id== dbPoint.get(pI).id && m_id== dbPoint.get(pJ).id ){
					out.setSubMatrix(2*i, 2*j,MatrixDouble.diagonal(2, g-1));
				}
				//l,n
				if(l_id== dbPoint.get(pI).id && n_id== dbPoint.get(pJ).id ){
					out.setSubMatrix(2*i, 2*j,MatrixDouble.diagonal(2, 1-g));
				}
				//m,k
				if(m_id== dbPoint.get(pI).id && k_id== dbPoint.get(pJ).id ){
					out.setSubMatrix(2*i, 2*j,MatrixDouble.diagonal(2, 1-g));
				}
				//m,l
				if(m_id== dbPoint.get(pI).id && l_id== dbPoint.get(pJ).id ){
					out.setSubMatrix(2*i, 2*j,MatrixDouble.diagonal(2, g-1));
				}
				//m,m
				if(m_id== dbPoint.get(pI).id && m_id== dbPoint.get(pJ).id ){
					//0
				}
				//m,n
				if(m_id== dbPoint.get(pI).id && n_id== dbPoint.get(pJ).id ){
					// 0
				}
				//n,k
				if(n_id== dbPoint.get(pI).id && k_id== dbPoint.get(pJ).id ){
					out.setSubMatrix(2*i, 2*j,MatrixDouble.diagonal(2, g-1));
				}
				//n,l
				if(n_id== dbPoint.get(pI).id && l_id== dbPoint.get(pJ).id ){
					out.setSubMatrix(2*i, 2*j,MatrixDouble.diagonal(2, 1-g));
				}
				//n,m
				if(n_id== dbPoint.get(pI).id && m_id== dbPoint.get(pJ).id ){
					// 0
				}
				//n,n
				if(n_id== dbPoint.get(pI).id && n_id== dbPoint.get(pJ).id ){
					// 0
				}
				j++;
			}
			i++;
		}

		return out;
	}
	@Override
	public boolean isHessianConst() {
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
		return param_id;
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {

		
	}
	@Override
	public double getNorm() {
		Vector vLK = ((Vector)dbPoint.get(l_id)).sub((Vector)dbPoint.get(k_id));
		Vector vNM = ((Vector)dbPoint.get(n_id)).sub((Vector)dbPoint.get(m_id));
		MatrixDouble md =getValue();
		return md.m[0][0]/(vLK.length()*vNM.length());
	}
}
