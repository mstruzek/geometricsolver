package com.mstruzek.msketch.matrix;

import com.mstruzek.msketch.Vector;

/**
 * Klasa reprezentuje macierz
 * + dodatkowe operacje
 *
 * @author root
 */
public class MatrixDouble2D implements MatrixDouble {

    /*** zmienna przechowujaca nasze elementy */
    public double[][] m = null;
    /*** szerokosc -ilosc kolumn */
    int columns;
    /*** wysokosc - ilosc wiersze */
    int rows;

    /**
     * Konstruktor macierzy
     *
     * @param columns szerokosc -ilosc kolumn
     * @param rows    wysokosc - ilosc wiersze
     */
    public MatrixDouble2D(int rows, int columns) {
        super();
        this.columns = columns;
        this.rows = rows;
        this.m = new double[rows][columns];
    }

    /**
     * Tworzy macierz na podstawie wektora
     *
     * @param vec        wektora
     * @param columnType true if coumn type [a b c]', false if row type [a b c]
     */
    public MatrixDouble2D(Vector vec, boolean columnType) {
        if (columnType) {
            this.columns = 1;
            this.rows = 2;
            this.m = new double[rows][columns];
            this.m[0][0] = vec.getX();
            this.m[1][0] = vec.getY();

        } else {
            this.columns = 2;
            this.rows = 1;
            this.m = new double[rows][columns];
            this.m[0][0] = vec.getX();
            this.m[0][1] = vec.getY();
        }
    }

    public static MatrixDouble identity(int size, double initial) {
        return MatrixDouble.identity(size, initial);
    }

    @Override
    public double get(int i, int j) {
        return this.m[i][j];
    }

    @Override
    public int width() {
        return columns;
    }

    @Override
    public int height() {
        return rows;
    }

    @Override
    public MatrixDouble add(MatrixDouble rhs) {
        assertEqualDimensions(this, rhs);
        for (int i = 0; i < this.m.length; i++) {
            for (int j = 0; j < this.m[i].length; j++) {
                this.m[i][j] = this.m[i][j] + rhs.get(i, j);
            }
        }
        return this;
    }

    @Override
    public MatrixDouble dot(double c) {
        for (int i = 0; i < height(); i++) {
            for (int j = 0; j < width(); j++) {
                this.m[i][j] *= c;
            }
        }
        return this;
    }

    @Override
    public MatrixDouble dotC(double c) {
        MatrixDouble mt = this.copy();
        for (int i = 0; i < height(); i++) {
            for (int j = 0; j < width(); j++) {
                mt.set(i, j, mt.get(i, j) * c);
            }
        }
        return mt;
    }


    @Override
    public MatrixDouble mult(MatrixDouble rhs) {
        if (this.width() != rhs.height()) throw new Error("Illegal dimension of right-hand side operand matrix");

        if (rhs instanceof SmallMatrixDouble && height() == rhs.height() && width() == rhs.width()) {
            SmallMatrixDouble srhs = (SmallMatrixDouble) rhs;
            double a00 = get(0, 0);
            double a01 = get(0, 1);
            double a10 = get(1, 0);
            double a11 = get(1, 1);
            SmallMatrixDouble mt = new SmallMatrixDouble(a00, a01, a10, a11);
            return mt.mult(rhs);
        } else {
            MatrixDouble2D mt = new MatrixDouble2D(rhs.width(), this.height());
            for (int i = 0; i < height(); i++) { /// this row
                for (int j = 0; j < rhs.width(); j++) { // other column
                    double acc = 0.0;
                    for (int k = 0; k < width(); k++) {     // other column
                        acc += this.m[i][k] * rhs.get(k, j);
                    }
                    mt.m[i][j] = acc;
                }
            }
            return mt;
        }
    }

    @Override
    public MatrixDouble copy() {
        MatrixDouble2D array = new MatrixDouble2D(this.m.length, this.m[0].length);
        for (int i = 0; i < array.m.length; i++)
            System.arraycopy(this.m[i], 0, array.m[i], 0, this.m[i].length);
        return array;
    }

    @Override
    public void set(int r, int c, double value) {
        this.m[r][c] = value;
    }

    @Override
    public void add(int r, int c, double value) {
        this.m[r][c] = this.m[r][c] + value;
    }

    @Override
    public MatrixDouble setSubMatrix(int firstRow, int firstColumn, MatrixDouble mt) {
        //sprawdzamy czy macierz wstawiana nie jest za duza
        if (this.height() < (firstRow + mt.height()) && this.width() < (firstColumn + mt.width())) {
            throw new IllegalStateException("dimension overflow");
        }

        if (mt instanceof ScalarMatrixDouble) {
            set(firstRow, firstColumn, ((ScalarMatrixDouble) mt).m);
            return this;
        }
        if (mt instanceof SmallMatrixDouble) {
            SmallMatrixDouble smt = (SmallMatrixDouble) mt;
            set(firstRow + 0, firstColumn + 0, smt.sm[0]);
            set(firstRow + 0, firstColumn + 1, smt.sm[1]);
            set(firstRow + 1, firstColumn + 0, smt.sm[2]);
            set(firstRow + 1, firstColumn + 1, smt.sm[3]);
            return this;
        }

        MatrixDouble2D mt2 = (MatrixDouble2D) mt;
        /// mozna wstawic
        for (int k = 0; k < mt.height(); k++) {
            System.arraycopy(mt2.m[k], 0, this.m[k + firstRow], firstColumn, mt2.m[k].length);
        }
        return this;
    }

    @Override
    public MatrixDouble addSubMatrix(int firstRow, int firstColumn, MatrixDouble mt) {
        if (this.height() < (firstRow + mt.height()) || this.width() < (firstColumn + mt.width())) {
            throw new Error("matrix dimension out of bounds");
        }
        if (mt instanceof SmallMatrixDouble) {
            SmallMatrixDouble smt = (SmallMatrixDouble) mt;
            add(firstRow + 0, firstColumn + 0, smt.sm[0]);
            add(firstRow + 0, firstColumn + 1, smt.sm[1]);
            add(firstRow + 1, firstColumn + 0, smt.sm[2]);
            add(firstRow + 1, firstColumn + 1, smt.sm[3]);
            return this;
        } else {
            MatrixDouble2D tmh = (MatrixDouble2D) mt;
            for (int i = 0; i < mt.height(); i++) {
                for (int j = 0; j < mt.width(); j++) {
                    m[i + firstRow][j + firstColumn] += tmh.m[i][j];
                }
            }
            return this;
        }
    }

    @Override
    public MatrixDouble setVector(int r, int c, Vector vector) {
        if (this.width() == 1) { /// row oriented
            set(r + 0, c, vector.getX());
            set(r + 1, c, vector.getY());
            return this;
        } else if (this.height() == 1) { /// column oriented
            set(r, c + 0, vector.getX());
            set(r, c + 1, vector.getY());
            return this;
        } else {
            throw new IllegalStateException("no implementation");
        }
    }

    @Override
    public MatrixDouble2D transpose() {
        MatrixDouble2D tm = new MatrixDouble2D(columns, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                tm.m[j][i] = this.m[i][j];
            }
        }
        return tm;
    }

    @Override
    public MatrixDouble reset(double value) {
        for (int i = 0; i < m.length; i++)
            for (int j = 0; j < m[i].length; j++)
                m[i][j] = value;
        return this;
    }


}
