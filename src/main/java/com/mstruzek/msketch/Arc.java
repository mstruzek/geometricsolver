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
public class Arc extends GeometricPrymitive{
	
	/** Licznik lukow */
	static int counter =0;
	/** numer kolejno utworzonego luku */
	int id = counter++;
	/** tablica wszystkich lukow*/
	static TreeMap<Integer,Arc> dbArc = new TreeMap<Integer,Arc> ();
	
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
	
	// FIXME -jakis blad na jakobianie i silach nie zbiega sie :(
	/** KOLEJNOSC INICJALIZACJI MA ZNACZENIE - PATRZ KONSTRUKTOR */
	
	/** Odleglosci pomiedzy wektorami poczatkowe */
	double d_a_p1,d_b_p1,d_p1_p2,d_p1_p3,d_p3_d,d_p2_c;
	/** wsp�czynnik do skalowania wzglednego dla wektorow*/
	double alfa = 1.5;
	
	/** Numery wiezow  powiazane z a,b*/
	int[] constraintsId = new int[5];
	
	
	/**
	 * Luk 
	 * @param p10 srodek okregu
	 * @param p20 pierwszy koniec luku
	 * @param p30 drugi koniec luku
	 */
	public Arc(Vector p10,Vector p20,Vector p30){
		
		// FreePoint(a,p1,b)
		Vector va,vb,vc,vd,v1,v2,v3;
		
		v1 = p10;
		v2 = p20;
		v3 = v1.add(v2.sub(v1).Rot(-90));
		va = v1.add(v1.sub(v2).dot(alfa));
		vb = v1.add(v2.sub(v1).Rot(90).dot(alfa));
		vc = v2.add(v2.sub(v1).dot(alfa));
		vd = v3.add(v2.sub(v1).Rot(-90).dot(alfa));
		
		a = new Point(va);
		b = new Point(vb);
		c = new Point(vc);
		d = new Point(vd);
		
		p1 = new Point(v1);
		p2 = new Point(v2);
		p3 = new Point(v3);

		
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
		//Naciag wstepny lepiej sie zbiegaja
		d_a_p1 	=Math.abs(p1.sub(a).length());
		d_b_p1	=Math.abs(p1.sub(b).length());
		d_p1_p2	=Math.abs(p2.sub(p1).length());
		d_p1_p3	=Math.abs(p3.sub(p1).length());
		d_p3_d	=Math.abs(d.sub(p3).length());
		d_p2_c	=Math.abs(c.sub(p2).length());

	}

	@Override
	public void recalculateControlPoints() {

		Vector va = (Vector) p1.sub(p2.sub(p1).dot(alfa));
		Vector vb = (Vector) p1.sub(p3.sub(p1).unit().dot(p2.sub(p1).length()).dot(alfa));
		Vector vc = (Vector) p2.add(p2.sub(p1).dot(alfa));
		Vector vd = (Vector) p3.add(p3.sub(p1).unit().dot(p2.sub(p1).length()).dot(alfa));
		Vector v3 = (Vector) p1.add(p3.sub(p1).unit().dot(p2.sub(p1).length()));
		
		
		a.setLocation(va.x,va.y);
		b.setLocation(vb.x,vb.y);
		c.setLocation(vc.x,vc.y);
		d.setLocation(vd.x,vd.y);	
		p3.setLocation(v3.x,v3.y);
		
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

		
		// K -mala sztywnosci
		MatrixDouble Kb = MatrixDouble.diagonal(Config.springStiffnessHigh,Config.springStiffnessHigh);
		MatrixDouble Ks = MatrixDouble.diagonal(Config.springStiffnessLow,Config.springStiffnessLow);
		
		MatrixDouble mKs = Ks.dotC(-1);
		MatrixDouble mKb = Kb.dotC(-1);
		MatrixDouble mKsKb = mKs.add(mKb);

		
		//a
		mt.addSubMatrix(0, 0, mKs).addSubMatrix(0, 8, Ks);
		//b
		mt.addSubMatrix(2, 2, mKs).addSubMatrix(2, 8, Ks);
		//c
		mt.addSubMatrix(4, 4, mKs).addSubMatrix(4,10, Ks);
		//d
		mt.addSubMatrix(6, 6, mKs).addSubMatrix(6,12, Ks);
		//p1
		mt.addSubMatrix(8, 0, Ks).addSubMatrix(8,2, Ks).addSubMatrix(8,8, mKsKb.dotC(2.0)).addSubMatrix(8,10, Kb).addSubMatrix(8,12, Kb);
		//p2
		mt.addSubMatrix(10,4, Ks).addSubMatrix(10,8, Kb).addSubMatrix(10,10,mKsKb);
		//p3
		mt.addSubMatrix(12,6, Ks).addSubMatrix(12,8, Kb).addSubMatrix(12,12,mKsKb);

		return mt;
	}

	@Override
	public int[] setAssociateConstraints() {
		ConstraintFixPoint fixPointa = new ConstraintFixPoint(a);
		ConstraintFixPoint fixPointb = new ConstraintFixPoint(b);
		ConstraintFixPoint fixPointc = new ConstraintFixPoint(c);
		ConstraintFixPoint fixPointd = new ConstraintFixPoint(d);
		ConstraintLinesSameLength sameLength = new ConstraintLinesSameLength(p1,p2,p1,p3);
		//ConstraintLinesSameLength sameLength2= new ConstraintLinesSameLength(p2,c,p3,d);
		//ConstraintLinesParallelism par1 = new ConstraintLinesParallelism(a,p1,p2,c);
		//ConstraintLinesParallelism par2 = new ConstraintLinesParallelism(b,p1,p3,d);

		constraintsId[0] = fixPointa.constraintId;
		constraintsId[1] = fixPointb.constraintId;
		constraintsId[2] = fixPointc.constraintId;
		constraintsId[3] = fixPointd.constraintId;
		constraintsId[4] = sameLength.constraintId;
		//constraintsId[5] = sameLength2.constraintId;
		//constraintsId[6] = par1.constraintId;
		//constraintsId[7] = par2.constraintId;
		
		return constraintsId;
	}

	@Override
	public MatrixDouble getForce() {
		MatrixDouble force = MatrixDouble.fill(14,1,0.0);

		Vector fap1 	= p1.sub(a).unit().dot(Config.springStiffnessLow).dot(p1.sub(a).length()-d_a_p1);	
		Vector fbp1 	= p1.sub(b).unit().dot(Config.springStiffnessLow).dot(p1.sub(b).length()-d_b_p1);
		Vector fp1p2 	= p2.sub(p1).unit().dot(Config.springStiffnessHigh).dot(p2.sub(p1).length()-d_p1_p2);
		Vector fp1p3 	= p3.sub(p1).unit().dot(Config.springStiffnessHigh).dot(p3.sub(p1).length()-d_p1_p3);
		Vector fp2c 	= c.sub(p2).unit().dot(Config.springStiffnessLow).dot(c.sub(p2).length()-d_p2_c);
		Vector fp3d 	= d.sub(p3).unit().dot(Config.springStiffnessLow).dot(d.sub(p3).length()-d_p3_d);
		
		force.addSubMatrix(0, 0, new MatrixDouble(fap1,true));
		force.addSubMatrix(2, 0, new MatrixDouble(fbp1 ,true));
		force.addSubMatrix(4, 0, new MatrixDouble(fp2c.dot(-1),true));
		force.addSubMatrix(6, 0, new MatrixDouble(fp3d.dot(-1),true));
		force.addSubMatrix(8, 0, new MatrixDouble(fp1p2.add(fp1p3).sub(fap1).sub(fbp1),true));
		force.addSubMatrix(10, 0, new MatrixDouble(fp2c.sub(fp1p2),true));	
		force.addSubMatrix(12, 0, new MatrixDouble(fp3d.sub(fp1p3),true));		
		
		return force;
	}

	@Override
	public int getNumOfPoints() {
		return 7; //a,b,c,d,p1,p2,p3
	}
	@Override
	public int getP1() {
		return p1.id;
	}

	@Override
	public int getP2() {
		return p2.id;
	}

	@Override
	public int getP3() {
		return p3.id;
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
		out[1] = b.getId();
		out[2] = c.getId();
		out[3] = d.getId();
		out[4] = p1.getId();
		out[5] = p2.getId();		
		out[6] = p3.getId();

		return out;
	}
	public static void main(String[] args) {
		
		Arc ln = new Arc(new Vector(0.0,0.0),new Vector(4.0,0.0),null);
		System.out.println(ln);
		ln.recalculateControlPoints();
		System.out.println(ln);
	}
}
