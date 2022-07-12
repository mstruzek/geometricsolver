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
	public static int parameterCounter =0;

	int id;
	/** tablica wszystkich parametrow*/
	public static TreeMap<Integer,Parameter> dbParameter  = new TreeMap<Integer,Parameter>();
	
	/**
	 * Glowny Konstruktor
	 * @param par parametr
	 */
	public Parameter(int id, double par) {
		if(id < parameterCounter) throw new RuntimeException("invalid object id");
		this.id = id;
		this.value = par;
		dbParameter.put(id,this);
	}

	public Parameter(double par) {
		this.value = par;
		this.id = parameterCounter++;
		dbParameter.put(id,this);
	}


	/**
	 * Pobierz parametr
	 * @return
	 */
	public double getValue() {
		return value;
		
	}

	public double getRadians() {
		return (Math.PI / 180) * value;
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
