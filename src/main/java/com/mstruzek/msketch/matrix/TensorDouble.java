package com.mstruzek.msketch.matrix;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import com.mstruzek.msketch.Vector;

import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * [ Concept ]:
 *
 * - instrukcje na malych macierzach 2x2 tylko w obrebie lokalnych skladowych - to prawie wszyskie funkcje z wyjatkiem.
 *
 * - drukujemy do impl macierzy tylko impl : [ addSubMatrix, setSubMatrix, setVector, transpose ]
 */
public interface TensorDouble {

    String DOUBLE_STR_FORMAT = " %11.2e";
//    String DOUBLE_STR_FORMAT = " %11.8f";
    String WIDEN_DOUBLE_STR_FORMAT = "%26s";


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
    TensorDouble plus(TensorDouble rhs);

    /**
     * Mnozezenie kazdego elementu macierzy przez skalar
     *
     * @param c skalar
     * @return this
     */
    TensorDouble mulitply(double c);

    /**
     * Mnozezenie kazdego elementu macierzy przez skalar
     * zwracana jest kopia , aktualna macierz niezmieniona
     *
     * @param c skalar
     * @return kopia macierzy aktualnej
     */
    TensorDouble multiplyC(double c);

    /**
     * Mnozezenie kazdego vectora macierzy przez odpowiadajacy  vector columnowy.
     *
     * !! do usuniecia !
     *
     * @param rhs prawy operand
     * @return
     */
    TensorDouble multiply(TensorDouble rhs);

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
    void plus(int r, int c, double value);


    /**
     * Create a sub view that will reference internal matrix implementation.
     *
     * @param row
     * @param column
     * @param height
     * @param width
     * @return
     */
    TensorDouble viewSpan(int row, int column, int height, int width);

    /**
     * Funkcja wstawia macierz mt na dana pozycje w akutalnej macierzy
     *
     * @param offsetRow poczatkowy wiersz
     * @param offsetCol poczatkowa kolumna
     * @param mt        macierz do wstawienia
     * @return this
     */
    TensorDouble setSubMatrix(int offsetRow, int offsetCol, TensorDouble mt);

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
    TensorDouble plusSubMatrix(int offsetRow, int offsetCol, TensorDouble mt);


    /**
     * Column oriented sub vector or row oriented vector.
     *
     * @param r
     * @param c
     * @param vector
     */
    TensorDouble setVector(int r, int c, Vector vector);

    /**
     * Transposes an mxn matrix into an nxm matrix. Each row of the input matrix becomes a column in the
     * output matrix.
     *
     * @return transposed cloned matrix.
     */
    TensorDouble transpose();

    /**
     * Reset matrix fill with constant value.
     *
     * @param value constant value that reset matrix to.
     * @return this matrix
     */
    TensorDouble reset(double value);

    /**
     * Internal implementation from Colt, MatrixDouble2D, SmallMatrixDouble, ScalarMatrixDouble.
     *
     * @param clazz
     * @param <T>
     * @return
     */
    <T> T unwrap(Class<T> clazz);


    default void assertEqualDimensions(TensorDouble left, TensorDouble right) {
        if (left.width() != right.width() || left.height() != right.height()) {
            throw new Error("Matrices must be of the same dimension");
        }
    }

    static TensorDouble scalar(double value) {
        return new ScalarTensorDouble(value);
    }

    /**
     * Make small columnar matrix 2x1 or row oriented matrix 1x2 from vector coordinates.
     * @param vector
     * @param columnar
     * @return
     */
    static TensorDouble smallMatrix(Vector vector, boolean columnar) {
        return new TensorDouble2D(vector, columnar);
    }

    static TensorDouble smallMatrix(double a00, double a01, double a10, double a11) {
        SmallTensorDouble mt = new SmallTensorDouble(a00, a01, a10, a11);
        return mt;
    }

    static TensorDouble identity(int size, double diag) {
        return DefaultDoubleMatrixCreator.INSTANCE.makeIdentity(size, diag);
//        return MatrixDoubleCreator.getInstance().makeIdentity(size, diag);
    }

    static TensorDouble diagonal(int size, double c) {
        return DefaultDoubleMatrixCreator.INSTANCE.makeDiagonal(size, c);
//        return MatrixDoubleCreator.getInstance().makeDiagonal(size, c);
    }

    static TensorDouble diagonal(double... diag) {
        return DefaultDoubleMatrixCreator.INSTANCE.makeDiagonal(diag);
//        return MatrixDoubleCreator.getInstance().makeDiagonal(diag);
    }

    static TensorDouble matrix2D(int rowSize, int colSize, double initValue) {
        return MatrixDoubleCreator.getInstance().makeMatrix2D(rowSize, colSize, initValue);
    }

    /**
     * Column Vector ! used mostly for righ-hand side matrix `b of equation  A*x = b
     *
     * @param rowSize
     * @param initValue
     * @return
     */
    static TensorDouble matrix1D(int rowSize, double initValue) {
        return MatrixDoubleCreator.getInstance().makeMatrix1D(rowSize, initValue);
    }


    /**
     * Wrap usually DenseDoubleMatrix1D into corresponding MatrixDouble adapter.
     *
     * @param delegate source vector
     * @return
     */
    static TensorDouble matrixDoubleFrom(DoubleMatrix1D delegate) {
        return MatrixDoubleCreator.getInstance().matrixDoubleFrom(delegate);
    }

    static TensorDouble matrixDoubleFrom(DoubleMatrix2D delegate) {
        return MatrixDoubleCreator.getInstance().matrixDoubleFrom(delegate);
    }



    /**
     * Small rotation  matrix around OZ axis.
     * [    cos(alfa)       -sin(alfa);
     * sin(alfa)        cos(alfa)      ]
     *
     * @param alfa degrees
     * @return
     */
    static TensorDouble rotation(double alfa) {
        double radians = Math.toRadians(alfa);
        double a00 = Math.cos(radians);
        double a01 = -1.0 * Math.sin(radians);
        double a10 = Math.sin(radians);
        double a11 = Math.cos(radians);
        SmallTensorDouble smd = new SmallTensorDouble(a00, a01, a10, a11);
        return smd;
    }

    static TensorDouble matrixR() {
        double a00 = 0.0;
        double a01 = -1.0;
        double a10 = 1.0;
        double a11 = 0.0;
        TensorDouble smd = TensorDouble.smallMatrix(a00, a01, a10, a11);
        return smd;
    }

    /**
     * Generates a string that holds organized version of a matrix.
     *
     * @return array string
     */
    static String toStringData(String format, TensorDouble tensorDouble) {
        StringBuffer str = new StringBuffer();
        for (int i = 0; i < tensorDouble.height(); i++) {
            String first = String.format(format, tensorDouble.getQuick(i, 0));
            str.append(first);
            for (int j = 1; j < tensorDouble.width(); j++) {
                String cell = String.format("," + format, tensorDouble.getQuick(i, j));
                str.append(cell);
            }
            if (i < tensorDouble.width() - 1 || tensorDouble.width()==1) str.append("\n");
        }
        return str.toString();
    }

    /**
     * @return A string of a nicely organized version of the matrix or array.
     */
    static String writeToString(TensorDouble tensorDouble) {
        StringBuffer str = new StringBuffer();
        str.append("\n")
            .append("MatrixDouble - ")
            .append(tensorDouble.height())
            .append("x")
            .append(tensorDouble.width())
            .append("****************************************\n")
            .append(IntStream.range(0, tensorDouble.width()/2).mapToObj(s -> String.format(WIDEN_DOUBLE_STR_FORMAT, s))
                .collect(Collectors.joining()))
            .append("\n")
            .append(toStringData(DOUBLE_STR_FORMAT, tensorDouble));
        return str.toString();
    }
}
