package pl.struzek.msketch;

import java.util.TreeMap;

import pl.struzek.msketch.matrix.MatrixDouble;
/**
 * Klasa FreePoint - czyli wolny Punkt 
 * 
 */
public class FreePoint extends GeometricPrymitive {
	/** Licznik freepointow */
	static int counter =0;
	/** numer kolejno utworzonego free pointa */
	int id = counter++;
	/** tablica wszystkich linii*/
	static TreeMap<Integer,FreePoint> dbFreePoint = new TreeMap<Integer,FreePoint>();
	
	/** fix control points -punkty kontrolne zafixowane */
	Point a = null; 
	Point b = null;
	/** dynamic points - punkty dynamiczne odpowiedzialne za przemiszczanie sie lini*/
	Point p1 =  null;
	
	/** Odleglosci pomiedzy wektorami poczatkowe */
	double d_a_p1,d_p1_b;
	/** wsp�czynnik do skalowania wzglednego dla wyznaczenia wektorow a,b*/
	double distance = 150.0;
	/** Kat rotacji wzgledem osi X ,dla wyznaczenie polozenia poczatkowego dla a,b - w stopniach*/
	double angle = Math.toRadians(40);
	
	/** Numery wiezow  powiazane z a,b*/
	int[] constraintsId = new int[2];	
	
	public FreePoint(Vector p) {
		super();
		// Punkty kontrolne wyznaczamy na podstawie distance,agnle : distance- to odleglosc wektora a i b od p1 
		//Kolejnosc inicjalizacji ma znaczenie
		a = new Point(p.sub(new Vector(distance*Math.cos(angle),distance*Math.sin(angle))));
		p1 = new Point(p);
		b = new Point(p.add(new Vector(distance*Math.cos(angle),distance*Math.sin(angle))));
		// przelicz odleglosci
		calculateDistance();
		//dodaj do globalnej bazy
		dbFreePoint.put(id,this);
		dbPrimitives.put(primitiveId,this);
		this.associateConstraintsId = setAssociateConstraints();
		// ustaw typ geometryczny
		type = GeometricPrymitiveType.FreePoint;
	}
	
	/** Funkcja oblicza dlugosci poczatkowe pomiedzy wektorami */
	private void calculateDistance() {

		d_a_p1 = Math.abs(p1.sub(a).length())*0.1;
		d_p1_b = Math.abs(b.sub(p1).length())*0.1;

	}
	public String toString(){
		String out = type + ""  + this.id + "*" + this.primitiveId + ": {";
		out+="a=" + a + ",";
		out+="p1=" + p1 + ",";
		out+="b=" + b + "}\n";
		return out;
	}

	@Override
	public void recalculateControlPoints() {
		Vector va =(p1.sub(new Vector(distance*Math.cos(angle),distance*Math.sin(angle))));
		Vector vb =(p1.add(new Vector(distance*Math.cos(angle),distance*Math.sin(angle))));
		a.setLocation(va.x,va.y);
		b.setLocation(vb.x,vb.y);
		// przelicz odleglosci
		calculateDistance();
		
		((ConstraintFixPoint)Constraint.dbConstraint.get(constraintsId[0])).setFixVector(va);
		((ConstraintFixPoint)Constraint.dbConstraint.get(constraintsId[1])).setFixVector(vb);
	
	}
	
	@Override
	public MatrixDouble getForceJacobian() {
		MatrixDouble mt = MatrixDouble.fill(6, 6, 0.0);
		/**
		 * k= I*k
		 * [ -ks    ks     0;
		 *    ks  -2ks   ks ;
		 *     0    ks   -ks];

		 */
		// K -mala sztywnosci
		MatrixDouble Ks = MatrixDouble.diagonal(Config.springStiffnessLow,Config.springStiffnessLow);
		MatrixDouble mKs = Ks.dotC(-1);
		

		mt.addSubMatrix(0, 0, mKs).addSubMatrix(0, 2, Ks);

		mt.addSubMatrix(2, 0, Ks).addSubMatrix(2, 2, mKs.dotC(2.0)).addSubMatrix(2, 4, Ks);
		mt.addSubMatrix(4, 2, Ks).addSubMatrix(4, 4, mKs);
		return mt;
	}

	@Override
	public int[] setAssociateConstraints() {
		ConstraintFixPoint fixPointa = new ConstraintFixPoint(a);
		ConstraintFixPoint fixPointb = new ConstraintFixPoint(b);
		constraintsId[0] = fixPointa.constraintId;
		constraintsId[1] = fixPointb.constraintId;
		return constraintsId;
	}

	@Override
	public MatrixDouble getForce() {
		// 8 = 4*2 (4 punkty kontrolne)
		MatrixDouble force = MatrixDouble.fill(6,1,0.0);
		
		//F12 - sily w sprezynach
		Vector f12 = p1.sub(a).unit().dot(Config.springStiffnessLow).dot(p1.sub(a).length()-d_a_p1);	
		//F23
		Vector f23 = b.sub(p1).unit().dot(Config.springStiffnessLow).dot(b.sub(p1).length()-d_p1_b);
		
		//F1 - silu na poszczegolne punkty
		force.addSubMatrix(0, 0, new MatrixDouble(f12,true));
		//F2
		force.addSubMatrix(2, 0, new MatrixDouble(f23.sub(f12),true));
		//F3
		force.addSubMatrix(4, 0, new MatrixDouble(f23.dot(-1),true));
		
		return force;
	}

	@Override
	public int getNumOfPoints() {
		return 3; //a,p1,p2
	}
	@Override
	public int getP1() {
		return p1.id;
	}

	@Override
	public int getP2() {
		return -1;
	}

	@Override
	public int getP3() {
		return -1;
	}

	@Override
	public int[] getAllPointsId() {
		int[] out = new int[3];
		
		out[0] = a.getId();
		out[1] = b.getId();
		out[2] = p1.getId();
		return out;
	}

	public static void main(String[] args) {
		
		FreePoint fp = new FreePoint(new Vector(1.0,1.0));
		FreePoint fp1 = new FreePoint(new Vector(1.0,1.0));
		System.out.println(fp);
		System.out.println(fp1);
		System.out.println(FreePoint.dbFreePoint);
	}

	@Override
	public int getA() {
		return a.id;
	}

	@Override
	public int getB() {
		return b.id;
	}

	@Override
	public int getC() {
		return -1;
	}

	@Override
	public int getD() {
		return -1;
	}
	
}
