package pl.struzek.msketch;

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
	static int counter =0;
	int id = counter++;
	/** tablica wszystkich parametrow*/
	public static TreeMap<Integer,Parameter> dbParameter  = new TreeMap<Integer,Parameter>();
	
	/**
	 * Glowny Konstruktor
	 * @param par parametr
	 */
	public Parameter(double par) {
		this.value = par;
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
	public static void main(String[] args) {
		
		Parameter pm = new Parameter(10.0);
		Parameter pm2 = new Parameter(30.0);
		System.out.println(pm);
		System.out.println(Parameter.dbParameter);

	}
	
}
