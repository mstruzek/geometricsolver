package com.mstruzek.msketch;

import java.util.TreeMap;

import com.mstruzek.msketch.matrix.MatrixDouble;

/**
 * Klasa abstrakcyjna dla podstawowych elementow geometrycznych jakie moga zostac
 * narysowane na ekranie : linia, luk, okrag,"wolny" punkt 
 *
 */
public abstract class GeometricPrymitive {
	/** licznik elementow podstawowych */
	static int primitiveCounter= 0;
	/** id danego elemntu podstawowego */
	int primitiveId =primitiveCounter++;
	/** tablica wszystkich elemntow*/
	public static TreeMap<Integer,GeometricPrymitive> dbPrimitives = new TreeMap<Integer,GeometricPrymitive>();
	/** Typ elementu */
	GeometricPrymitiveType type =null;
	/** tablica przechowujaca powiazane wiezy dla punktow kontrolnych np : a,b,c*/
	int[] associateConstraintsId = null;
	
	/** Przelicza na nowo pozycje punktow kontrolnych */
	public abstract void recalculateControlPoints();
	
	/** Funkcja zwraca jakobian si� - czyli macierz szytnowsci Fq */
	public abstract MatrixDouble getForceJacobian();
	/** Funkcja zwraca wartosc sil w sprezynach dla poszczegolnych punkt�w w danym {@link GeometricPrymitive} */
	public abstract MatrixDouble getForce();
	
	/** Pobierz wszystkie punkty powiazane z dana figura */
	public abstract int[] getAllPointsId();
	
	/** Funkcja  ustawia wiezy poczatkowe -czyli rejestruje w bazie wiezow, np : wiez FixPoint na Point[a,b,c] */
	public abstract int[] setAssociateConstraints();
	
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
	public GeometricPrymitiveType getType() {
		return type;
	}

	/** Funkcja zwraca pelny jakobian sil dla wszystkich elementow geometrycznych */
	public static MatrixDouble getAllForceJacobian(){
		
		int size = Point.dbPoint.size()*2;
		
		MatrixDouble out = MatrixDouble.fill(size, size, 0.0);
		
		int currentRowCol=0;
		for(Integer i:dbPrimitives.keySet()){
			out.addSubMatrix(currentRowCol, currentRowCol, dbPrimitives.get(i).getForceJacobian());
			currentRowCol+=dbPrimitives.get(i).getNumOfPoints()*2;
		}
		return out;
	}
	
	/** Funkcja zwraca wartosc sil w sprezynach dla wszystkich punktow */
	public static MatrixDouble getAllForce(){
		
		int size = Point.dbPoint.size()*2;
		
		MatrixDouble out = MatrixDouble.fill(size, 1, 0.0);
		
		int currentRowCol=0;
		for(Integer i:dbPrimitives.keySet()){
			out.addSubMatrix(currentRowCol, 0, dbPrimitives.get(i).getForce());
			currentRowCol+=dbPrimitives.get(i).getNumOfPoints()*2;
		}
		return out;		
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
	
}
