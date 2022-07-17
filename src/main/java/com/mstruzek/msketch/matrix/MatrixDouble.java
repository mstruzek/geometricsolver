package com.mstruzek.msketch.matrix;

import com.mstruzek.msketch.Vector;

import java.util.Arrays;
import java.util.stream.Collectors;

/**
 * [ Concept ]:
 *
 * - instrukcje na malych macierzach 2x2 tylko w obrebie lokalnych skladowych - to prawie wszyskie funkcje z wyjatkiem.
 *
 * - drukujemy do impl macierzy tylko impl : [ addSubMatrix, setSubMatrix, setVector, transpose ]
 *
 */
public interface MatrixDouble {

    String DOUBLE_STR_FORMAT = "%11.2f";
    String WIDEN_DOUBLE_STR_FORMAT = "%23s ";

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
    double getQuick(int i, int j);

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
     * Mnozezenie kazdego vectora macierzy przez odpowiadajacy  vector columnowy.
     *
     * !! do usuniecia !
     * @param rhs prawy operand
     * @return
     */
    MatrixDouble mult(MatrixDouble rhs);

    /**
     * Set value at corresponding coordinates.
     *
     * @param r     row column
     * @param c
     * @param value
     */
    void setQuick(int r, int c, double value);

    /**
     * Add value at corresponding coordinates.
     *
     * @param r     row column
     * @param c
     * @param value
     */
    void add(int r, int c, double value);


    /**
     * Create a sub view that will reference internal matrix implementation.
     * @param row
     * @param column
     * @param height
     * @param width
     * @return
     */
    MatrixDouble viewSpan(int row, int column, int height, int width);

    /**
     * Funkcja wstawia macierz mt na dana pozycje w akutalnej macierzy
     *
     * @param offsetRow poczatkowy wiersz
     * @param offsetCol poczatkowa kolumna
     * @param mt        macierz do wstawienia
     * @return this
     */
    MatrixDouble setSubMatrix(int offsetRow, int offsetCol, MatrixDouble mt);

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
     * @param offsetRow poczatkowy wiersz
     * @param offsetCol poczatkowa kolumna
     * @param mt        macierz do wstawienia
     * @return this matrix
     */
    MatrixDouble addSubMatrix(int offsetRow, int offsetCol, MatrixDouble mt);


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

    /**
     * Internal implementation from Colt, MatrixDouble2D, SmallMatrixDouble, ScalarMatrixDouble.
     * @param clazz
     * @return
     * @param <T>
     */
    <T> T unwrap(Class<T> clazz);


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
        return DefaultDoubleMatrixCreator.INSTANCE.makeIdentity(size, diag);
//        return MatrixDoubleCreator.getInstance().makeIdentity(size, diag);
    }

    static MatrixDouble diagonal(int size, double c) {
        return DefaultDoubleMatrixCreator.INSTANCE.makeDiagonal(size, c);
//        return MatrixDoubleCreator.getInstance().makeDiagonal(size, c);
    }

    static MatrixDouble diagonal(double... diag) {
        return DefaultDoubleMatrixCreator.INSTANCE.makeDiagonal(diag);
//        return MatrixDoubleCreator.getInstance().makeDiagonal(diag);
    }

    static MatrixDouble matrix2D(int rowSize, int colSize, double initValue) {
        return MatrixDoubleCreator.getInstance().makeMatrix2D(rowSize, colSize, initValue);
    }

    /**
     * Column Vector ! used mostly for righ-hand side matrix `b of equation  A*x = b
     * @param rowSize
     * @param initValue
     * @return
     */
    static MatrixDouble matrix1D(int rowSize, double initValue) {
        return MatrixDoubleCreator.getInstance().makeMatrix1D(rowSize, initValue);
    }

    /**
     * Small rotation  matrix around OZ axis.
     * [    cos(alfa)       -sin(alfa);
     *      sin(alfa)        cos(alfa)      ]
     * @param alfa degrees
     * @return
     */
    static MatrixDouble rotation(double alfa) {
        double radians = Math.toRadians(alfa);
        double a00 = Math.cos(radians);
        double a01 = -1.0 * Math.sin(radians);
        double a10 = Math.sin(radians);
        double a11 = Math.cos(radians);
        SmallMatrixDouble smd = new SmallMatrixDouble(a00, a01, a10, a11);
        return smd;
    }

    static MatrixDouble matrixR() {
        double a00 = 0.0;
        double a01 = -1.0;
        double a10 = 1.0;
        double a11 = 0.0;
        MatrixDouble smd = MatrixDouble.smallMatrix(a00, a01, a10, a11);
        return smd;
    }

    /**
     * Generates a string that holds organized version of a matrix.
     *
     * @return array string
     */
    default String toString(String format) {
        StringBuffer str = new StringBuffer();
        for (int i = 0; i < this.height(); i++) {
            String first = String.format(format + " ", this.getQuick(i, 0));
            str.append(first);
            for (int j = 1; j < this.width(); j++) {
                String cell = String.format("," + format + " ", this.getQuick(i, j));
                str.append(cell);
            }
            if (i < this.width() - 1) str.append("\n");
        }
        return str.toString();
    }

    /**
     * @return A string of a nicely organized version of the matrix or array.
     */
    default <T> String toString(T... titles) {
        StringBuffer str = new StringBuffer();
        str
            .append("\n")
            .append("MatrixDouble - ")
            .append(this.height())
            .append("x")
            .append(this.width())
            .append("****************************************\n");

        if (titles.length != 0) {
            str
                .append(Arrays.stream(titles).map(s -> String.format(WIDEN_DOUBLE_STR_FORMAT, s.toString()))
                    .collect(Collectors.joining()))
                .append("\n");
        }

        str.append(toString(DOUBLE_STR_FORMAT));
        return str.toString();
    }

    default String toStringU() {
        StringBuffer str = new StringBuffer();
        str.append(toString(DOUBLE_STR_FORMAT));
        return str.toString();
    }
}
