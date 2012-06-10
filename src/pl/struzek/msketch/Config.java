package pl.struzek.msketch;

/**
 * Klasa konfirugaracyjna przechowujaca rozne dane wspoldzielone
 * np. sztywnosci sprezyny pomiedzy punktami 
 * @author root
 *
 */
public class Config {
	
	/** tymczasowa szytnowsc sprezyny */
	static double springStiffness = 2;
	/** Sztywnosc sprezyny wysoka-  pomiedzy punktami wolnymi czyli "p*"*/
	static double springStiffnessHigh = 29;
	/** Sztywnosc sprezyny niska - glownie dla polaczenia pomiedzy punktem 
	 * zafiksowanym "{a,b}" i nie zafiksowanym "p*" */
	static double springStiffnessLow = 1;
	
	
	/**
	 * Nakladka na funkcje atan2 zwraca kat z przedzialu 0-2pi
	 * @param y -wspolrzedne punktu
	 * @param x -wspolrzedne punktu
	 * @return kat w radianach
	 */
	public static double atan2(double y,double x){
		double fi =-1* Math.atan2(y, x);
		if(fi<0){
			
			fi = Math.PI*2+fi;
		}
		return fi;
	}

}
