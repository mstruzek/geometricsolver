package pl.struzek.msketch;

import java.util.TreeMap;

import Jama.Matrix;

import pl.struzek.msketch.matrix.MatrixDouble;

/**
 * Klasa reprezentujaca wiez typu polaczenie
 * dwoch punktow(wektorow uogulnionych) PointK-PoinL=0;
 * @author root
 *
 */
public class ConstraintConect2Points extends Constraint {

	/** Point K-id */
	int k_id;
	/** Point L-id */
	int l_id;

	/**
	 * Konstruktor wiezu
	 * @param K - zrzutowany Point na Vector
	 * @param L - Vector w ktorym bedzie zafiksowany K
	 */
	public ConstraintConect2Points(Point K,Point L) {
		super(GeometricConstraintType.Conect2Points);
		//pobierz id
		k_id = ((Point)K).id;
		l_id=((Point)L).id;
		dbConstraint.put(constraintId,this);
	}
	
	
	public String toString(){
		MatrixDouble out = getValue(Point.dbPoint, Parameter.dbParameter);
		double norm = Matrix.constructWithCopy(out.getArray()).norm1();
		return "Constraint-Conect2Points" + constraintId + "*s" + size() + " = " + norm + " { K =" + Point.dbPoint.get(k_id) + "  , L = " + Point.dbPoint.get(l_id) + " } \n";
	}

	@Override
	public MatrixDouble getJacobian(TreeMap<Integer, Point> dbPoints,TreeMap<Integer, Parameter> dbParameter) {
		//macierz 2 wierszowa
		MatrixDouble out = MatrixDouble.fill(2,dbPoints.size()*2,0.0);
		//zerujemy cala macierz + wstawiamy na odpowiednie miejsce Jacobian wiezu
		int j=0;
		for(Integer i:dbPoints.keySet()){
			out.m[0][j*2]= 0.0 ;out.m[0][j*2+1] = 0.0;
			out.m[1][j*2]= 0.0 ;out.m[1][j*2+1] = 0.0;			
			
			//a tu wstawiamy macierz dla tego wiezu
			if(k_id==dbPoints.get(i).id){
				//macierz jednostkowa = I
				out.m[0][j*2]= 1.0 ;out.m[0][j*2+1] = 0.0;
				out.m[1][j*2]= 0.0 ;out.m[1][j*2+1] = 1.0;				
			}	
			if(l_id==dbPoints.get(i).id){
				// = -I
				out.m[0][j*2]= -1.0 ;out.m[0][j*2+1] = 0.0;
				out.m[1][j*2]= 0.0  ;out.m[1][j*2+1] = -1.0;				
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
		
		return new MatrixDouble(((Vector)dbPoints.get(k_id)).sub((Vector)dbPoints.get(l_id)),true);
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
		return l_id;
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
		Point pn2 = new Point(new Vector(4.0,5.0));
		ConstraintConect2Points conectPoint = new ConstraintConect2Points(pn,pn2);
		
		Point pn3 =new Point(new Vector(3.0,4.0));
		ConstraintFixPoint fixPoint2 = new ConstraintFixPoint(pn3);
		
		Constraint cn = fixPoint2;
		System.out.println(conectPoint.getJacobian(Point.dbPoint, Parameter.dbParameter));
		System.out.println(Constraint.dbConstraint );

	}


	@Override
	public double getNorm(TreeMap<Integer, Point> dbPoints,TreeMap<Integer, Parameter> dbParameter) {
		MatrixDouble md = getValue(dbPoints, dbParameter);
        return Math.sqrt(md.m[0][0]*md.m[0][0]+md.m[1][0]*md.m[1][0]);

	}

}
