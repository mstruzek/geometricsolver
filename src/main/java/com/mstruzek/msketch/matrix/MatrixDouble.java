package com.mstruzek.msketch.matrix;

import com.mstruzek.msketch.Vector;

import java.util.Arrays;
import java.util.stream.Collectors;

public interface MatrixDouble {

    /**
     * Number of columns
     *
     * @return
     */
    int width();

    /**
     * Number of rows
     *
     * @return
     */
    int height();

    /**
     * Get value at corresponding coordinates.
     *
     * @param i - row
     * @param j - column
     * @return double value
     */
    double get(int i, int j);

    /**
     * Adds a matrix to another matrix, which should be of the same dimension.
     *
     * @returns The resulting matrix , actual matrix this
     */
    MatrixDouble add(MatrixDouble rhs);

    /**
     * Mnozezenie kazdego elementu macierzy przez skalar
     *
     * @param c skalar
     * @return this
     */
    MatrixDouble dot(double c);

    /**
     * Mnozezenie kazdego elementu macierzy przez skalar
     * zwracana jest kopia , aktualna macierz niezmieniona
     *
     * @param c skalar
     * @return kopia macierzy aktualnej
     */
    MatrixDouble dotC(double c);

    /**
     * Mnozezenie kazdego vectora macierzy na odpowiadajacy  vector columnowy.
     *
     * @param rhs prawy operand
     * @return
     */
    MatrixDouble mult(MatrixDouble rhs);

    /**
     * Generate a copy of a matrix
     *
     * @return A copy of this matrix
     */
    MatrixDouble copy();

    /**
     * Set value at corresponding coordinates.
     *
     * @param r     row column
     * @param c
     * @param value
     */
    void set(int r, int c, double value);

    /**
     * Add value at corresponding coordinates.
     *
     * @param r     row column
     * @param c
     * @param value
     */
    void add(int r, int c, double value);

    /**
     * Funkcja wstawia macierz mt na dana pozycje w akutalnej macierzy
     *
     * @param firstRow    poczatkowy wiersz
     * @param firstColumn poczatkowa kolumna
     * @param mt          macierz do wstawienia
     * @return this
     */
    MatrixDouble setSubMatrix(int firstRow, int firstColumn, MatrixDouble mt);

    /**
     * Funkcja dodaje macierz mt do aktualnej macierzy i zwraca kopie , macierz this niezmieniona
     * A = [a11 a12 a13;
     * a21 a22 a23]
     * B [b11 b12;
     * b21 b22;
     * A.addSubMatrix(0,1,B)= C
     * [a11 a12+b11 a13+b12;
     * a21 a22+b21 a23+b22]
     *
     * @param firstRow    poczatkowy wiersz
     * @param firstColumn poczatkowa kolumna
     * @param mt          macierz do wstawienia
     * @return this matrix
     */
    MatrixDouble addSubMatrix(int firstRow, int firstColumn, MatrixDouble mt);


    /**
     * Column oriented sub vector or row oriented vector.
     *
     * @param r
     * @param c
     * @param vector
     */
    MatrixDouble setVector(int r, int c, Vector vector);

    /**
     * Transposes an mxn matrix into an nxm matrix. Each row of the input matrix becomes a column in the
     * output matrix.
     *
     * @return transposed cloned matrix.
     */
    MatrixDouble transpose();

    /**
     * Reset matrix fill with constant value.
     *
     * @param value constant value that reset matrix to.
     * @return this matrix
     */
    MatrixDouble reset(double value);


    default void assertEqualDimensions(MatrixDouble left, MatrixDouble right) {
        if (left.width() != right.width() || left.height() != right.height()) {
            throw new Error("Matrices must be of the same dimension");
        }
    }

    static MatrixDouble scalar(double value) {
        return new ScalarMatrixDouble(value);
    }

    static MatrixDouble smallMatrix(double a00, double a01, double a10, double a11) {
        SmallMatrixDouble mt = new SmallMatrixDouble(a00, a01, a10, a11);
        return mt;
    }

    static MatrixDouble identity(int size, double diag) {
        return MatrixDoubleCreator.getCreatorInstance().makeIdentity(size, diag);
    }

    static MatrixDouble diagonal(int size, double c) {
        return MatrixDoubleCreator.getCreatorInstance().makeDiagonal(size, c);
    }

    static MatrixDouble diagonal(double... diag) {
        return MatrixDoubleCreator.getCreatorInstance().makeDiagonal(diag);
    }

    static MatrixDouble matrix2D(int rowSize, int colSize, double initValue) {
        return MatrixDoubleCreator.getCreatorInstance().makeMatrix2D(rowSize, colSize, initValue);
    }

    static MatrixDouble matrix1D(int colSize, double initValue) {
        return MatrixDoubleCreator.getCreatorInstance().makeMatrix1D(colSize, initValue);
    }

    static MatrixDouble matrix1Dtr(int rowSize, double initValue) {
        return MatrixDoubleCreator.getCreatorInstance().makeMatrix1Dtr(rowSize, initValue);
    }

    static MatrixDouble rotation2d(double alfa) {
        return MatrixDoubleCreator.getCreatorInstance().makeRotation2d(alfa);
    }

    static MatrixDouble matrixR() {
        return MatrixDouble.smallMatrix(0.0, -1.0, 1.0, 0.0);
    }

    /**
     * Dodaje do siebie Macierze w ten sposub ze kazda kolejna macierz
     * jest pod druga w kolumnie , macierze powinny miec ta sama ilosc kolumn ale nie koniecznie
     * <p>
     * M = mergeByColumn(A,B,C);
     * M= [ A;
     * B;
     * C];
     *
     * @param MD - lista macierzy do polaczenia
     * @return
     */
    static MatrixDouble mergeByColumn(MatrixDouble... MD) {
        if (Arrays.stream(MD).anyMatch(matrixDouble -> !(matrixDouble instanceof MatrixDouble2D))) {
            throw new IllegalStateException("invalid state, argument matrix is required to be MatrixDouble2D type");
        }
        int maxRows = 0;
        int maxColumns = 0;
        for (int i = 0; i < MD.length; i++) {
            maxRows += MD[i].height();
            if (MD[i].width() > maxColumns) maxColumns = MD[i].width();
        }
        MatrixDouble MT = MatrixDouble.matrix2D(maxRows, maxColumns, 0.0);
        int currentRow = 0;
        for (int i = 0; i < MD.length; i++) {
            for (int j = 0; j < MD[i].height(); j++) { //j - numer wiersza w danej macierzy
                System.arraycopy(((MatrixDouble2D) MD[i]).m[j], 0, ((MatrixDouble2D) MT).m[currentRow + j], 0, ((MatrixDouble2D) MD[i]).m[j].length);
            }
            currentRow += MD[i].height();
        }
        return MT;
    }


    /**
     * Generates a string that holds a nicely organized version of a matrix or array.
     * Uses format specifier, e.g. "%5.3f", "%11.1E", ...
     *
     * @return A string of a nicely organized version of the matrix or array.
     */
    default String toString(String format) {
        StringBuffer buffer = new StringBuffer();
        for (int i = 0; i < this.height(); i++) {
            for (int j = 0; j < this.width() - 1; j++) {
                buffer.append(String.format(format + " ", this.get(i, j)));
            }
            buffer.append(String.format(format, this.get(i, this.width() - 1)));
            if (i < this.height() - 1) {
                buffer.append("\n");
            }
        }
        return buffer.toString();
    }

    /**
     * @return A string of a nicely organized version of the matrix or array.
     */
    default <T> String toString(T... columnTitles) {
        StringBuffer buffer = new StringBuffer();
        buffer.append("\n").append("MatrixDouble - ").append(this.height()).append("x").append(this.width())
            .append("\n****************************************\n");
        if (columnTitles.length != 0) {
            buffer.append(Arrays.stream(columnTitles).map(s -> String.format("%23s ", s.toString())).collect(Collectors.joining()))
                .append("\n");
        }
        buffer.append(toString("%11.2f"));
        return buffer.toString();
    }

}
