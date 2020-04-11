package com.mstruzek.sparsematrixsolver;

/**
 * Interfejs reprezentuj�cy podstawowy model macierz
 * Z tego interfejsu wywodza sie DenseMatrix oraz SparseMatrix
 * 
 * @author root
 *
 */
public interface MatrixData {

	/**
	 * Mnozenie macierzy przez wektor 
	 * c = A * b 
	 * @param out wektor wyjsciowy "c"
	 * @param in wektor kolumnowy "b"
	 * @return wektor kolumnowy "c"
	 * @param startRow poczatkowy wiersz wektora "b"
	 * @param startColumn poczatek w kolumnie
	 */
	void multiply( BasicVector out,BasicVector in,int startRow,int startColumn);
	
	/**
	 * Pomnoz kazdy element macierzy przez skalar
	 * @param in skalar
	 */
	void multiply(double in);
	
	/**
	 * Odwroc macierz
	 * @return kopia macierzy
	 */
	MatrixData transposeC();
	
	void add(int row,int col,double val);
	/**
	 * Ustaw wartosc na danje pozycji
	 * @param row wiersz
	 * @param col kolumna 
	 * @param val
	 */
	void set(int row,int col,double val);
	
	/**
	 * Poierz wartosc na danej pozycji
	 * @param row wiersz	
	 * @param col kolumna
	 * @return
	 */
	double get(int row,int col);

	/**
	 * Pobierz szerokosc okna macierzy
	 * @return
	 */
	int getWidth();
	
	/**
	 * Pobierz wysokosc okna macierzy
	 * @return
	 */
	int getHeight();
}
