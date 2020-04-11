package com.mstruzek.msketch;

import java.util.TreeMap;

import com.mstruzek.msketch.matrix.MatrixDouble;

/**
 * Luk geometryczny 
 *  - sklada sie z srodka okregu i dwoch punktow krancowych luku
 *  W naszym wypatku to konstrukcja zlozona z linii (czyli 2 punkty na luku)
 *  + FreePoint -srodek okregu + wiez rownej dlugosci pomiedzy punktami a srodkiem
 *  
 *
 */
//FIXME - UWAGA przy wstawianiu LUKU nalezy pamietac aby promien znajdowal sie na symetralnej pomiedzy punktami p2,p3
public class ArcOld extends GeometricPrymitive{
	
	/** Licznik lukow */
	static int counter =0;
	/** numer kolejno utworzonego luku */
	int id = counter++;
	/** tablica wszystkich lukow*/
	static TreeMap<Integer, ArcOld> dbArc = new TreeMap<Integer, ArcOld> ();
	
	/** fix control points -punkty kontrolne zafixowane */
	Point a = null; 
	Point b = null;
	Point c = null;
	Point d = null;
	/** dynamic points - punkty dynamiczne odpowiedzialne za przemiszczanie sie lini*/
	Point p1 =  null;
	Point p2 =  null;
	Point p3 =  null;
	
	// FreePoint(a,p1,b) , Line(c,p2,p3,d) 
	
	
	/** KOLEJNOSC INICJALIZACJI MA ZNACZENIE - PATRZ KONSTRUKTOR */
	
	/** Odleglosci pomiedzy wektorami poczatkowe */
	double d_a_p1,d_p1_b,d_c_p2,d_p2_p3,d_p3_d;
	/** wsp�czynnik do skalowania wzglednego dla wektorow*/
	double alfa = 1.0;
	
	/** Numery wiezow  powiazane z a,b*/
	int[] constraintsId = new int[5];
	
	/** Mnoznik sily */
	int dS =10;
	
	/**
	 * Luk 
	 * @param p10 srodek okregu
	 * @param p20 pierwszy koniec luku
	 * @param p30 drugi koniec luku
	 */
	public ArcOld(Vector p10,Vector p20,Vector p30){
		
		// FreePoint(a,p1,b)
		a=new Point(p10.dot(2).sub(p20));
		p1=new Point(p10);
		b=new Point(p20);
		
		//  Line(c,p2,p3,d)
		Vector a= p20.sub(p10);
		c=new Point(p10.add(a.Rot(-90).dot(3)));
		p2=new Point(p10.add(a.Rot(-90)));
		p3=new Point(p10.add(a.Rot(90)));
		d=new Point(p10.add(a.Rot(90).dot(3)));
		
		calculateDistance();
		dbArc.put(id,this);
		dbPrimitives.put(primitiveId,this);
		this.associateConstraintsId = setAssociateConstraints();
		// ustaw typ geometryczny
		type = GeometricPrymitiveType.Arc;
	}

	public String toString(){
		String out = type + "" + this.id + "*" + this.primitiveId + ": {";
		out+="a =" + a  + ",";
		out+="p1=" + p1 + ",";
		out+="b =" + b  + ",";
		out+="c =" + c  + ",";
		out+="p2=" + p2 + ",";
		out+="p3=" + p3 + ",";
		out+="d =" + d  + "}\n";		
		return out;
	}
	
	/** Funkcja oblicza dlugosci poczatkowe pomiedzy wektorami */
	private void calculateDistance() {
		// FreePoint(a,p1,b)
		d_a_p1 = Math.abs(p1.sub(a).length());
		d_p1_b = Math.abs(b.sub(p1).length());
		
		// Line(c,p2,p3,d)
		d_c_p2 =  Math.abs(p2.sub(c).length());
		d_p2_p3 =  Math.abs(p3.sub(p2).length());
		d_p3_d = Math.abs(p3.sub(d).length());
	}

	@Override
	public void recalculateControlPoints() {

		Vector pr =(Vector)(p3.sub(p2).Rot(-90).dot(0.5));
		Vector va = (Vector)p1.add(pr);
		Vector vb = (Vector)p1.sub(pr);
		//Line
		Vector vc =(Vector)p2.sub(p3.sub(p2));
		Vector vd =(Vector)p3.add(p3.sub(p2));
		
		
		a.setLocation(va.x,va.y);
		b.setLocation(vb.x,vb.y);
		c.setLocation(vc.x,vc.y);
		d.setLocation(vd.x,vd.y);	
		
		calculateDistance();
		
		//uaktulaniamy wiezy
		((ConstraintFixPoint)Constraint.dbConstraint.get(constraintsId[0])).setFixVector(va);
		((ConstraintFixPoint)Constraint.dbConstraint.get(constraintsId[1])).setFixVector(vb);
		((ConstraintFixPoint)Constraint.dbConstraint.get(constraintsId[2])).setFixVector(vc);
		((ConstraintFixPoint)Constraint.dbConstraint.get(constraintsId[3])).setFixVector(vd);
		
	}


	@Override
	public MatrixDouble getForceJacobian() {
		MatrixDouble mt = MatrixDouble.fill(14, 14, 0.0);
		/** Free Point
		 * k= I*k
		 * [ -ks    ks     0;
		 *    ks  -2ks   ks ;
		 *     0    ks   -ks];

		 */
		/** Line -shift = 6
		 * k= I*k
		 * [ -ks    ks      0  	  0;
		 *    ks  -ks-kb   kb  	  0;
		 *     0    kb   -ks-kb   ks;
		 *     0  	 0     ks    -ks];
		 */
		
		// K -mala sztywnosci
		MatrixDouble Ksp = MatrixDouble.diagonal(Config.springStiffnessHigh*dS,Config.springStiffnessHigh*dS);
		MatrixDouble mKsp = Ksp.dotC(-1);
		
		MatrixDouble Ks = MatrixDouble.diagonal(Config.springStiffnessLow,Config.springStiffnessLow);
		MatrixDouble mKs = Ks.dotC(-1);
		
		MatrixDouble Kb = MatrixDouble.diagonal(Config.springStiffnessHigh,Config.springStiffnessHigh);
		// -Ks-Kb
		MatrixDouble Ksb = Ks.dotC(-1).addSubMatrix(0, 0, Kb.dotC(-1));

		//FREEPOINT
		mt.addSubMatrix(0, 0, mKsp).addSubMatrix(0, 2, Ksp);
		mt.addSubMatrix(2, 0, Ksp).addSubMatrix(2, 2, Ksp.dotC(2.0)).addSubMatrix(2, 4, Ksp);
		mt.addSubMatrix(4, 2, Ksp).addSubMatrix(4, 4, mKsp);
		
		//LINE
		int s = 6;//przesuniecie
		mt.addSubMatrix(0+s, 0+s, Ks.dotC(-1)).addSubMatrix(0+s, 2+s, Ks);
		mt.addSubMatrix(2+s, 0+s, Ks).addSubMatrix(2+s, 2+s, Ksb.dotC(2.0)).addSubMatrix(2+s, 4+s, Kb);
		mt.addSubMatrix(4+s, 2+s, Kb).addSubMatrix(4+s, 4+s, Ksb).addSubMatrix(4+s, 6+s, Ks);
		mt.addSubMatrix(6+s, 4+s, Ks).addSubMatrix(6+s, 6+s, Ks.dotC(-1));
		return mt;
	}

	@Override
	public int[] setAssociateConstraints() {
		ConstraintFixPoint fixPointa = new ConstraintFixPoint(a);
		ConstraintFixPoint fixPointb = new ConstraintFixPoint(b);
		ConstraintFixPoint fixPointc = new ConstraintFixPoint(c);
		ConstraintFixPoint fixPointd = new ConstraintFixPoint(d);
		ConstraintLinesSameLength sameLength = new ConstraintLinesSameLength(p1,p2,p1,p3);
		constraintsId[0] = fixPointa.constraintId;
		constraintsId[1] = fixPointb.constraintId;
		constraintsId[2] = fixPointc.constraintId;
		constraintsId[3] = fixPointd.constraintId;
		constraintsId[4] = sameLength.constraintId;
		return constraintsId;
	}

	@Override
	public MatrixDouble getForce() {
		MatrixDouble force = MatrixDouble.fill(14,1,0.0);

		//F12 - sily w sprezynach
		Vector f12 = p1.sub(a).unit().dot(Config.springStiffnessHigh*dS).dot(p1.sub(a).length()-d_a_p1);	
		//F23
		Vector f23 = b.sub(p1).unit().dot(Config.springStiffnessHigh*dS).dot(b.sub(p1).length()-d_p1_b);
		
		//FREEPOINT
		//F1 - silu na poszczegolne punkty
		force.addSubMatrix(0, 0, new MatrixDouble(f12,true));
		//F2
		force.addSubMatrix(2, 0, new MatrixDouble(f23.sub(f12),true));
		//F3
		force.addSubMatrix(4, 0, new MatrixDouble(f23.dot(-1),true));
		
		
		//LINE
		Vector fcp2 = p2.sub(c).unit().dot(Config.springStiffnessLow).dot(p2.sub(c).length()-d_c_p2);	
		//F23
		Vector fp2p3 = p3.sub(p2).unit().dot(Config.springStiffnessHigh).dot(p3.sub(p2).length()-d_p2_p3);
		
		Vector fp3d = d.sub(p3).unit().dot(Config.springStiffnessLow).dot(d.sub(p3).length()-d_p3_d);		

		force.addSubMatrix(6, 0, new MatrixDouble(fcp2,true));
		force.addSubMatrix(7, 0, new MatrixDouble(fp2p3.sub(fcp2),true));
		force.addSubMatrix(10, 0, new MatrixDouble(fp3d.sub(fp2p3),true));	
		force.addSubMatrix(12, 0, new MatrixDouble(fp3d.dot(-1),true));		
		
		return force;
	}

	@Override
	public int getNumOfPoints() {
		return 7; //a,b,c,d,p1,p2,p3
	}
	@Override
	public int getP1() {
		// TODO Auto-generated method stub
		return p1.id;
	}

	@Override
	public int getP2() {
		// TODO Auto-generated method stub
		return p2.id;
	}

	@Override
	public int getP3() {
		// TODO Auto-generated method stub
		return p3.id;
	}
	
	@Override
	public int getA() {
		// TODO Auto-generated method stub
		return a.id;
	}

	@Override
	public int getB() {
		return b.id;
	}

	@Override
	public int getC() {
		return c.id;
	}

	@Override
	public int getD() {
		return d.id;
	}
	
	@Override
	public int[] getAllPointsId() {
		int[] out = new int[7];
		
		out[0] = a.getId();
		out[1] = p1.getId();
		out[2] = b.getId();
		out[3] = c.getId();
		out[4] = p2.getId();
		out[5] = p3.getId();		
		out[6] = d.getId();

		return out;
	}
	public static void main(String[] args) {
		
		Arc ln = new Arc(new Vector(0.0,0.0),new Vector(4.0,0.0),null);
		System.out.println(ln);
		ln.recalculateControlPoints();
		System.out.println(ln);
	}
}
