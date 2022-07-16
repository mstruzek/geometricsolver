package com.mstruzek.sparsematrixsolver;

/**
 * Klasa reprezentuje kotwice dzieki ktorej
 * "zaczepimy" pozycje danej macierzy w macierzystej macierzy
 *
 * @author root
 */
public class MatrixAnchor {

    /**
     * pozycja w wierszu macierzy rodzica
     */
    int rowPosition;
    /**
     * pozycja w kolumnie macierzy rodzica
     */
    int columnPosition;

    /**
     * link do macierzy zawierajacej dane
     */
    MatrixData matrixData;

    /**
     * Konstruktor podstawowy
     *
     * @param rowPosition
     * @param columnPosition
     * @param matrixData
     */
    public MatrixAnchor(int rowPosition, int columnPosition,
                        MatrixData matrixData) {
        super();
        this.rowPosition = rowPosition;
        this.columnPosition = columnPosition;
        this.matrixData = matrixData;
    }


}
