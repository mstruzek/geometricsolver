package com.mstruzek.msketch;

import java.util.TreeMap;

import com.mstruzek.msketch.matrix.MatrixDouble;

/** 
 * Klasa reprezentujaca zafiksowana linie,
 * zawiera wersory osi X,Y;
 *
 */
public class FixLine extends GeometricPrymitive{

	/** Licznik lini */
	static int counter =0;
	/** numer kolejno utworzonej lini */
	int id = counter++;
	/** tablica wszystkich linii*/
	static TreeMap<Integer,FixLine> dbFixLine = new TreeMap<Integer,FixLine>();
	
	/** fix control points */
	Vector a = null; 
	Vector b = null;
	
	//Pierwotne Linie naszego szkicownika
	static FixLine X = new FixLine(new Vector(0.0,0.0),new Vector(100.0,0.0));
	static FixLine Y = new FixLine(new Vector(0.0,0.0),new Vector(0.0,100.0));
	
	public FixLine(Vector a1,Vector b1){
		a=new Vector(a1);
		b=new Vector(b1);
		dbFixLine.put(id,this);
		
		//dbPrimitives.put(primitiveId,this); //nie dodajemy do primitiwov
		// ustaw typ geometryczny
		type = GeometricPrymitiveType.FixLine;
	}
	public String toString(){
		String out = type + ""  + id + "*" + this.primitiveId + ": {";
		out+="a = " + a + ",";
		out+="b = " + b + "}\n";
		return out;
	}

	@Override
	public void recalculateControlPoints() {
		
	}

	public static void main(String[] args) {
		
		//FixLine ln = new FixLine(new Vector(0.0,0.0),new Vector(4.0,3.0));

		//System.out.println(ln);
		System.out.println(FixLine.dbFixLine);

	}
	@Override
	public MatrixDouble getForceJacobian() {
		//poniewaz nie ma punktow kontrolnych to brak macierzy 
		return null;
	}
	@Override
	public int[] setAssociateConstraints() {
		//brak punktow kontrolnych
		return null;
	}
	@Override
	public MatrixDouble getForce() {
		//nic nie ma - brak sprezyn
		return null;
	}
	@Override
	public int getNumOfPoints() {
		return 0; //brak pointow
	}
	@Override
	public int getP1() {
		return	-1;
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
	public int getA() {
		return -1;
	}

	@Override
	public int getB() {
		return -1;
	}

	@Override
	public int getC() {
		return -1;
	}

	@Override
	public int getD() {
		return -1;
	}
	
	@Override
	public int[] getAllPointsId() {
		return null;
	}
}
