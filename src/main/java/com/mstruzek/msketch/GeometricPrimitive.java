package com.mstruzek.msketch;

import java.util.Set;
import java.util.TreeMap;

import com.mstruzek.msketch.matrix.MatrixDouble;

/**
 * Klasa abstrakcyjna dla podstawowych elementow geometrycznych jakie moga zostac
 * narysowane na ekranie : linia, luk, okrag,"wolny" punkt 
 *
 */
public abstract class GeometricPrimitive{

	/** licznik elementow podstawowych */
	public static int primitiveCounter= 0;

	/** id danego elemntu podstawowego */
	protected int primitiveId;
	/** Typ elementu */
	protected GeometricPrimitiveType type =null;

	/** tablica wszystkich elemntow*/
	public static TreeMap<Integer,GeometricPrimitive> dbPrimitives = new TreeMap<Integer,GeometricPrimitive>();


	/** tablica przechowujaca powiazane wiezy dla punktow kontrolnych np : a,b,c*/
	int[] constraints = null;

	public GeometricPrimitive(int primitiveId, GeometricPrimitiveType geometricPrimitiveType){
		this.primitiveId = primitiveId;
		this.type = geometricPrimitiveType;
		if(primitiveId >=0) dbPrimitives.put(primitiveId,this);
	}

	/** Przelicza na nowo pozycje punktow kontrolnych */
	public abstract void recalculateControlPoints();
	
	/** Funkcja zwraca jakobian si� - czyli macierz szytnowsci Fq */



	enum MtProcedure {

		AddSubMtx,	/// assignment with addition operator 	`+=`

		SetSubMtx	/// assignment operator 				`=`
	};

	/**
	 * @param procedure MtProcedure
	 * @param dest matrix
	 * @param firstRow
	 * @param firstCol
	 * @return
	 */
	public abstract void getJacobian(MtProcedure procedure, MatrixDouble dest, int firstRow, int firstCol);

	public abstract MatrixDouble getForce(MtProcedure procedure, MatrixDouble dest, int firstRow, int firstCol);


	public abstract MatrixDouble getJacobian();
	/** Funkcja zwraca wartosc sil w sprezynach dla poszczegolnych punkt�w w danym {@link GeometricPrimitive} */
	public abstract MatrixDouble getForce();

	/** Pobierz wszystkie punkty powiazane z dana figura */
	public abstract int[] getAllPointsId();
	
	/** Funkcja  ustawia wiezy poczatkowe -czyli rejestruje w bazie wiezow, np : wiez FixPoint na Point[a,b,c] */
	public abstract void setAssociateConstraints(Set<Integer> skipIds);


	/** Funkcja zwraca ilosc punktow w danym elemencie geometrycznym */
	public abstract int getNumOfPoints();
	
	/** Funkcja zwraca punktu p1,p2,p3 - potrzebne do wyswietlania*/
	public abstract int getP1();
	public abstract int getP2();
	public abstract int getP3();
	public abstract int getA();
	public abstract int getB();
	public abstract int getC();
	public abstract int getD();
	
	/**
	 * Zwraca id figury
	 * @return
	 */
	public int getPrimitiveId() {
		return primitiveId;
	}

	/**
	 * Zwraca rodzaj figury
	 * @return
	 */
	public GeometricPrimitiveType getType() {
		return type;
	}

	/** Funkcja zwraca pelny jakobian sil dla wszystkich elementow geometrycznych */
	public static MatrixDouble getAllJacobianForces(){
		MatrixDouble out = MatrixDouble.fill(Point.dbPoint.size()*2, Point.dbPoint.size()*2, 0.0);
		
		int currentRowCol=0;
		for(Integer i:dbPrimitives.keySet()){
			out.addSubMatrix(currentRowCol, currentRowCol, dbPrimitives.get(i).getJacobian());
			currentRowCol+=dbPrimitives.get(i).getNumOfPoints()*2;
		}
		return out;
	}
	
	/** Funkcja zwraca wartosc sil w sprezynach dla wszystkich punktow */
	public static MatrixDouble getAllForce(){
		MatrixDouble mt = MatrixDouble.fill(Point.dbPoint.size()*2, 1, 0.0);
		
		int currentRowCol=0;
		for(Integer i:dbPrimitives.keySet()){
			mt.addSubMatrix(currentRowCol, 0, dbPrimitives.get(i).getForce());
			currentRowCol+=dbPrimitives.get(i).getNumOfPoints()*2;
		}
		return mt;
	}
	
	//FIXME - trzeba jakos kontrolowac rozklad sily (glownie dla punktow kontrolnych a,b,c), jezeli sila jest zbyt duza na nowo poprzeliczac punkty
	//public abstract void relaxForces();

	/**
	 * Usun primiteve o danym id
	 * @param id - firgury
	 */
	public static void remove(int id){
		dbPrimitives.remove(id);
	}

	public int[] associateConstraintsId(){
		return constraints;
	}

	public static int nextId() {
		return primitiveCounter++;
	}

}
