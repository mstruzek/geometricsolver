package com.mstruzek.msketch;

import java.util.TreeMap;

/**
 * Klasa reprezentuje Parametr - dlugosc lub wartosc kat (w stopniach)
 * @author root
 *
 */
public class Parameter {
	
	/** parametr*/
	double value;
	
	/** Licznik parametrow*/
	static int counter=0;
	int id;
	/** tablica wszystkich parametrow*/
	public static TreeMap<Integer,Parameter> dbParameter  = new TreeMap<Integer,Parameter>();
	
	/**
	 * Glowny Konstruktor
	 * @param par parametr
	 */
	public Parameter(int id, double par) {
		if(id < counter) throw new RuntimeException("invalid object id");
		this.id = id;
		this.value = par;
		dbParameter.put(id,this);
	}

	public Parameter(double par) {
		this.value = par;
		this.id = counter++;
		dbParameter.put(id,this);
	}
	/**
	 * Pobierz parametr
	 * @return
	 */
	public double getValue() {
		return value;
		
	}
	/**
	 * Ustaw/zmien parametr
	 * @param param
	 */
	public void setValue(double param) {
		this.value = param;
	}
	public String toString(){
		return "param"+id+ " = " + value + "\n";
		
	}
	
	public int getId() {
		return id;
	}
	
}
