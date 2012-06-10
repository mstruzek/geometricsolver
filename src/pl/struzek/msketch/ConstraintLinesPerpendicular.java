package pl.struzek.msketch;

import java.util.TreeMap;

import Jama.Matrix;
import pl.struzek.msketch.matrix.MatrixDouble;
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
	 * @param K
	 * @param L
	 * @param M
	 * @param N
	 */
	public ConstraintLinesPerpendicular(Point K, Point L ,Vector M,Vector N){
		super(GeometricConstraintType.LinesPerpendicular);
		
		k_id = K.id;
		l_id = L.id;
		if((M instanceof Point) && ( N instanceof Point)){
			m_id =((Point)M).id;
			n_id = ((Point)N).id;
		}else{
			m=M;
			n= N;			
		}
		
		dbConstraint.put(constraintId,this);
	}
	public String toString(){
		MatrixDouble out = getValue(Point.dbPoint, Parameter.dbParameter);
		double norm = Matrix.constructWithCopy(out.getArray()).norm1();
		if(m==null && n==null) return "Constraint-LinesPerpendicular" + constraintId + "*s" + size() + " = " + norm  + " { K =" + Point.dbPoint.get(k_id) + "  ,L =" + Point.dbPoint.get(l_id) + " ,M =" + Point.dbPoint.get(m_id) + ",N =" + Point.dbPoint.get(n_id) + "} \n";
		else{
			return "Constraint-LinesPerpendicular" + constraintId + "*s" + size() + " = " + norm  + " { K =" + Point.dbPoint.get(k_id) + "  ,L =" + Point.dbPoint.get(l_id) + " ,vecM =" + m + ",vecN =" + n + "} \n";
		}
		
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
					Vector v1 = ((Vector)dbPoints.get(m_id)).sub((Vector)dbPoints.get(n_id));
					out.m[0][j*2]= v1.x ;out.m[0][j*2+1] = v1.y;
				}	
				if(l_id==dbPoints.get(i).id){
					Vector v1 = ((Vector)dbPoints.get(m_id)).sub((Vector)dbPoints.get(n_id));
					out.m[0][j*2]= -v1.x ;out.m[0][j*2+1] = -v1.y;				
				}
				//a tu wstawiamy macierz dla tego wiezu
				if(m_id==dbPoints.get(i).id){
					Vector v1 = ((Vector)dbPoints.get(k_id)).sub((Vector)dbPoints.get(l_id));
					out.m[0][j*2]= v1.x ;out.m[0][j*2+1] = v1.y;
				}	
				if(n_id==dbPoints.get(i).id){
					Vector v1 = ((Vector)dbPoints.get(k_id)).sub((Vector)dbPoints.get(l_id));
					out.m[0][j*2]= -v1.x ;out.m[0][j*2+1] = -v1.y;			
				}
				j++;
			}			
		}else{
			for(Integer i:dbPoints.keySet()){
				
				//a tu wstawiamy macierz dla tego wiezu
				if(k_id==dbPoints.get(i).id){
					Vector v1 = m.sub(n);
					out.m[0][j*2]= v1.x ;out.m[0][j*2+1] = v1.y;
				}	
				if(l_id==dbPoints.get(i).id){
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
	public MatrixDouble getValue(TreeMap<Integer, Point> dbPoints,TreeMap<Integer, Parameter> dbParameter) {
		Vector out = new Vector(dbPoints.get(k_id));
		out = out.sub((Vector)dbPoints.get(l_id));
		//out =out.unit();
		
		MatrixDouble mt = new MatrixDouble(1,1);
		
		if((m==null) && ( n ==null)){
			mt.m[0][0] = out.dot(((Vector)dbPoints.get(m_id)).sub((Vector)dbPoints.get(n_id)));
		}else{
			mt.m[0][0] = out.dot(m.sub(n));
		}
		return mt;
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
						out.m[2*i][2*j] = 1.0; out.m[2*i+1][2*j+1] = 1.0;
					}
					//k,n
					if(k_id==dbPoints.get(i).id && n_id==dbPoints.get(j).id ){
						out.m[2*i][2*j] = -1.0; out.m[2*i+1][2*j+1] = -1.0;
					}
					//l,m
					if(l_id==dbPoints.get(i).id && m_id==dbPoints.get(j).id ){
						out.m[2*i][2*j] = -1.0; out.m[2*i+1][2*j+1] = -1.0;
					}
					//l,n
					if(l_id==dbPoints.get(i).id && n_id==dbPoints.get(j).id ){
						out.m[2*i][2*j] = 1.0; out.m[2*i+1][2*j+1] = 1.0;
					}
					//m,k
					if(m_id==dbPoints.get(i).id && k_id==dbPoints.get(j).id ){
						out.m[2*i][2*j] = 1.0; out.m[2*i+1][2*j+1] = 1.0;
					}
					//m,l
					if(m_id==dbPoints.get(i).id && l_id==dbPoints.get(j).id ){
						out.m[2*i][2*j] = -1.0; out.m[2*i+1][2*j+1] = -1.0;
					}
					//n,k
					if(n_id==dbPoints.get(i).id && k_id==dbPoints.get(j).id ){
						out.m[2*i][2*j] = -1.0; out.m[2*i+1][2*j+1] = -1.0;
					}
					//n,l
					if(n_id==dbPoints.get(i).id && l_id==dbPoints.get(j).id ){
						out.m[2*i][2*j] = 1.0; out.m[2*i+1][2*j+1] = 1.0;
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
		Point pn3 = new Point(0.0,0.0);
		Point pn4 = new Point(1.0,10.0);
		//Vector pn3 = new Vector(1.0,1.0);
		//Vector pn4 = new Vector(1.0,2.0);
		ConstraintLinesPerpendicular cn = new ConstraintLinesPerpendicular(pn2,pn1,pn4,pn3);
		System.out.println(Constraint.dbConstraint );
		System.out.println(cn.getNorm(Point.dbPoint, Parameter.dbParameter));
		System.out.println(cn.getValue(Point.dbPoint, Parameter.dbParameter));
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
