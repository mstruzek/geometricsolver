package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Klasa reprezentujaca wiez prosopadlosci pomiedzy dwoma wektorami
 * skladajacymi sie z 4 punktow 
 * lub 2 punktow i jednego FixLine
 * @author root
 *
 */
public class ConstraintLinesPerpendicular extends Constraint {

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
	
	/**
     * Konstruktor pomiedzy 4 punktami lub
     * 2 punktami i FixLine(czyli 2 wektory)
     * rownanie tego wiezu to (K-L)'*(M-N) = 0
     * iloczyn skalarny
     *
     * @param constId
     * @param K
     * @param L
     * @param M
     * @param N
     */
	public ConstraintLinesPerpendicular(int constId,Point K,Point L ,Vector M,Vector N){
		super(constId, GeometricConstraintType.LinesPerpendicular, true);
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
		MatrixDouble out = getValue();
		double norm = Matrix.constructWithCopy(out.getArray()).norm1();
		if(m==null && n==null) return "Constraint-LinesPerpendicular" + constraintId + "*s" + size() + " = " + norm  + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) + "} \n";
		else{
			return "Constraint-LinesPerpendicular" + constraintId + "*s" + size() + " = " + norm  + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,vecM =" + m + ",vecN =" + n + "} \n";
		}
		
	}
	
	@Override
	public MatrixDouble getJacobian() {
		//macierz 2 wierszowa
		MatrixDouble out = MatrixDouble.fill(1,dbPoint.size()*2,0.0);
		//zerujemy cala macierz + wstawiamy na odpowiednie miejsce Jacobian wiezu
		int j=0;
		if((m==null) && ( n ==null)){
			for(Integer i:dbPoint.keySet()){
				
				//a tu wstawiamy macierz dla tego wiezu
				if(k_id==dbPoint.get(i).id){
					Vector v1 = ((Vector)dbPoint.get(m_id).Vector()).sub((Vector)dbPoint.get(n_id));
					out.m[0][j*2]= v1.x ;out.m[0][j*2+1] = v1.y;
				}	
				if(l_id==dbPoint.get(i).id){
					Vector v1 = ((Vector)dbPoint.get(m_id)).sub((Vector)dbPoint.get(n_id));
					out.m[0][j*2]= -v1.x ;out.m[0][j*2+1] = -v1.y;				
				}
				//a tu wstawiamy macierz dla tego wiezu
				if(m_id==dbPoint.get(i).id){
					Vector v1 = ((Vector)dbPoint.get(k_id)).sub((Vector)dbPoint.get(l_id));
					out.m[0][j*2]= v1.x ;out.m[0][j*2+1] = v1.y;
				}	
				if(n_id==dbPoint.get(i).id){
					Vector v1 = ((Vector)dbPoint.get(k_id)).sub((Vector)dbPoint.get(l_id));
					out.m[0][j*2]= -v1.x ;out.m[0][j*2+1] = -v1.y;			
				}
				j++;
			}			
		}else{
			for(Integer i:dbPoint.keySet()){
				
				//a tu wstawiamy macierz dla tego wiezu
				if(k_id==dbPoint.get(i).id){
					Vector v1 = m.sub(n);
					out.m[0][j*2]= v1.x ;out.m[0][j*2+1] = v1.y;
				}	
				if(l_id==dbPoint.get(i).id){
					Vector v1 = m.sub(n);
					out.m[0][j*2]= -v1.x ;out.m[0][j*2+1] = -v1.y;				
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
	public MatrixDouble getValue() {
		Vector out = new Vector(dbPoint.get(k_id));
		out = out.sub((Vector)dbPoint.get(l_id));
		//out =out.unit();
		
		MatrixDouble mt = new MatrixDouble(1,1);
		
		if((m==null) && ( n ==null)){
			mt.m[0][0] = out.dot(((Vector)dbPoint.get(m_id)).sub((Vector)dbPoint.get(n_id)));
		}else{
			mt.m[0][0] = out.dot(m.sub(n));
		}
		return mt;
	}

	@Override
	public MatrixDouble getHessian() {

		//macierz NxN
		MatrixDouble out = MatrixDouble.fill(dbPoint.size()*2,dbPoint.size()*2,0.0);


		if((m==null) && ( n ==null)){
			//same punkty
			int i = 0;
			for(Integer vI:dbPoint.keySet()){ //wiersz
				int j=0;
				for(Integer vJ:dbPoint.keySet()){ //kolumna
					//wstawiamy I,-I w odpowiednie miejsca
					//k,m
					if(k_id==dbPoint.get(vI).id && m_id==dbPoint.get(vJ).id ){
						out.m[2*i][2*j] = 1.0; out.m[2*i+1][2*j+1] = 1.0;
					}
					//k,n
					if(k_id==dbPoint.get(vI).id && n_id==dbPoint.get(vJ).id ){
						out.m[2*i][2*j] = -1.0; out.m[2*i+1][2*j+1] = -1.0;
					}
					//l,m
					if(l_id==dbPoint.get(vI).id && m_id==dbPoint.get(vJ).id ){
						out.m[2*i][2*j] = -1.0; out.m[2*i+1][2*j+1] = -1.0;
					}
					//l,n
					if(l_id==dbPoint.get(vI).id && n_id==dbPoint.get(vJ).id ){
						out.m[2*i][2*j] = 1.0; out.m[2*i+1][2*j+1] = 1.0;
					}
					//m,k
					if(m_id==dbPoint.get(vI).id && k_id==dbPoint.get(vJ).id ){
						out.m[2*i][2*j] = 1.0; out.m[2*i+1][2*j+1] = 1.0;
					}
					//m,l
					if(m_id==dbPoint.get(vI).id && l_id==dbPoint.get(vJ).id ){
						out.m[2*i][2*j] = -1.0; out.m[2*i+1][2*j+1] = -1.0;
					}
					//n,k
					if(n_id==dbPoint.get(vI).id && k_id==dbPoint.get(vJ).id ){
						out.m[2*i][2*j] = -1.0; out.m[2*i+1][2*j+1] = -1.0;
					}
					//n,l
					if(n_id==dbPoint.get(vI).id && l_id==dbPoint.get(vJ).id ){
						out.m[2*i][2*j] = 1.0; out.m[2*i+1][2*j+1] = 1.0;
					}
					j++;
				}
				i++;
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

		Point pn1 = new Point(Point.nextId(),0.0,0.0);
		Point pn2 = new Point(Point.nextId(),10.0,0.0);
		Point pn3 = new Point(Point.nextId(),0.0,0.0);
		Point pn4 = new Point(Point.nextId(),1.0,10.0);
		//Vector pn3 = new Vector(1.0,1.0);
		//Vector pn4 = new Vector(1.0,2.0);
		ConstraintLinesPerpendicular cn = new ConstraintLinesPerpendicular(Constraint.nextId(),pn2,pn1,pn4,pn3);
		System.out.println(Constraint.dbConstraint );
		System.out.println(cn.getNorm());
		System.out.println(cn.getValue());
	}
	@Override
	public double getNorm() {
		
		double val = getValue().m[0][0];
		
		Vector out = new Vector(((Vector)dbPoint.get(k_id)).sub((Vector)dbPoint.get(l_id)));
		val=val/out.length();
		
		if((m==null) && ( n ==null)){
			val=val/((Vector)dbPoint.get(m_id)).sub((Vector)dbPoint.get(n_id)).length();
		}else{
			val=val/(m.sub(n)).length();
		}
		
		return val;
	}

}
