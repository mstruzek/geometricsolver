package com.mstruzek.msketch.matrix;

import com.mstruzek.msketch.Vector;

/**
 * Klasa reprezentuje macierz  dodatkowe operacje , takze w postaci wspoldzielonego widoku !
 */
public class MatrixDouble2D implements MatrixDouble {

    /*** zmienna przechowujaca nasze elementy */
    public double[][] m = null;
    /*** szerokosc -ilosc kolumn */
    int columns;
    /*** wysokosc - ilosc wiersze */
    int rows;
    /*** base row offset - view concept */
    int rowOffset;
    /***  base column offset - view concept */
    int colOffset;

    /**
     * Konstruktor macierzy
     *
     * @param columns szerokosc -ilosc kolumn
     * @param rows    wysokosc - ilosc wiersze
     */
    public MatrixDouble2D(int rows, int columns) {
        this.rowOffset = 0;
        this.colOffset = 0;
        this.columns = columns;
        this.rows = rows;
        this.m = new double[rows][columns];
    }

    /**
     * Shared span with offset into matrix .
     *
     * @param rows
     * @param columns
     */
    public MatrixDouble2D(double[][] shared, int rowOffset, int colOffset, int rows, int columns) {
        this.rowOffset = rowOffset;
        this.colOffset = colOffset;
        this.rows = rows;
        this.columns = columns;
        this.m = shared;
    }

    /**
     * Tworzy macierz na podstawie wektora
     *
     * @param vec        wektora
     * @param columnType true if column type, false otherwise row type
     */
    public MatrixDouble2D(Vector vec, boolean columnType) {
        this.rowOffset = 0;
        this.colOffset = 0;
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

    @Override
    public double getQuick(int i, int j) {
        return this.m[rowOffset + i][colOffset + j];
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
        double value;
        for (int i = 0; i < height(); i++) {
            for (int j = 0; j < width(); j++) {
                this.m[rowOffset + i][colOffset + j] += rhs.getQuick(i, j);
            }
        }
        return this;
    }

    @Override
    public MatrixDouble dot(double c) {
        for (int i = 0; i < height(); i++) {
            for (int j = 0; j < width(); j++) {
                this.m[rowOffset + i][colOffset + j] *= c;
            }
        }
        return this;
    }

    @Override
    public MatrixDouble dotC(double c) {
        MatrixDouble mt = new MatrixDouble2D(width(), height());
        for (int i = 0; i < height(); i++) {
            for (int j = 0; j < width(); j++) {
                mt.setQuick(i, j, mt.getQuick(i, j) * c);
            }
        }
        return mt;
    }


    @Override
    public MatrixDouble mult(MatrixDouble rhs) {
        if (this.width() != rhs.height()) throw new Error("Illegal dimension of right-hand side operand matrix");

        if (rhs instanceof SmallMatrixDouble && height() == rhs.height() && width() == rhs.width()) {
            double a00 = getQuick(0, 0);
            double a01 = getQuick(0, 1);
            double a10 = getQuick(1, 0);
            double a11 = getQuick(1, 1);
            SmallMatrixDouble mt = new SmallMatrixDouble(a00, a01, a10, a11);
            return mt.mult(rhs);
        } else {
            MatrixDouble2D mt = new MatrixDouble2D(rhs.width(), this.height());
            for (int i = 0; i < height(); i++) { /// this row
                for (int j = 0; j < rhs.width(); j++) { // other column
                    double acc = 0.0;
                    for (int k = 0; k < width(); k++) {     // other column
                        acc += this.getQuick(i, k) * rhs.getQuick(k, j);
                    }
                    mt.setQuick(i, j, acc);
                }
            }
            return mt;
        }
    }

    @Override
    public void setQuick(int r, int c, double value) {
        this.m[rowOffset + r][colOffset + c] = value;
    }

    @Override
    public void add(int r, int c, double value) {
        this.m[rowOffset + r][colOffset + c] += value;
    }

    @Override
    public MatrixDouble viewSpan(int rowOffset, int colOffset, int height, int width) {
        /*
         * first level base tensor sharing policy !
         */
        if(rowOffset + height > height()) {
            throw new Error("out ouf bound offset request");
        }
        if(colOffset + width > width()) {
            throw new Error("out ouf bound offset request");
        }
        MatrixDouble2D md = new MatrixDouble2D(this.m, rowOffset, colOffset, height, width );
        return md;
    }

    @Override
    public MatrixDouble setSubMatrix(int offsetRow, int offsetCol, MatrixDouble mt) {
        //sprawdzamy czy macierz wstawiana nie jest za duza
        if (this.height() < (offsetRow + mt.height()) && this.width() < (offsetCol + mt.width())) {
            throw new IllegalStateException("dimension overflow");
        }

        if (mt instanceof ScalarMatrixDouble) {
            setQuick(offsetRow, offsetCol, ((ScalarMatrixDouble) mt).m);
            return this;
        }
        if (mt instanceof SmallMatrixDouble) {
            SmallMatrixDouble smt = (SmallMatrixDouble) mt;
            setQuick(offsetRow + 0, offsetCol + 0, smt.sm[0]);
            setQuick(offsetRow + 0, offsetCol + 1, smt.sm[1]);
            setQuick(offsetRow + 1, offsetCol + 0, smt.sm[2]);
            setQuick(offsetRow + 1, offsetCol + 1, smt.sm[3]);
            return this;
        }

        MatrixDouble2D mtd = (MatrixDouble2D) mt;
        /// mozna wstawic
        for (int k = 0; k < mt.height(); k++) {
            System.arraycopy(mtd.m[k], 0, this.m[k + rowOffset + offsetRow], colOffset + offsetCol, mtd.m[k].length);
        }
        return this;
    }

    @Override
    public MatrixDouble addSubMatrix(int offsetRow, int offsetCol, MatrixDouble mt) {
        if (this.height() < (offsetRow + mt.height()) || this.width() < (offsetCol + mt.width())) {
            throw new Error("matrix dimension out of bounds");
        }
        if (mt instanceof SmallMatrixDouble) {
            SmallMatrixDouble smt = (SmallMatrixDouble) mt;
            add(offsetRow + 0, offsetCol + 0, smt.sm[0]);
            add(offsetRow + 0, offsetCol + 1, smt.sm[1]);
            add(offsetRow + 1, offsetCol + 0, smt.sm[2]);
            add(offsetRow + 1, offsetCol + 1, smt.sm[3]);
            return this;
        } else {
            MatrixDouble2D tmh = (MatrixDouble2D) mt;
            for (int i = 0; i < mt.height(); i++) {
                for (int j = 0; j < mt.width(); j++) {
                    m[i + offsetRow][j + offsetCol] += tmh.m[i][j];
                }
            }
            return this;
        }
    }

    @Override
    public MatrixDouble setVector(int r, int c, Vector vector) {
        if (this.width() == 1) { /// row oriented
            setQuick(r + 0, c, vector.getX());
            setQuick(r + 1, c, vector.getY());
            return this;
        } else if (this.height() == 1) { /// column oriented
            setQuick(r, c + 0, vector.getX());
            setQuick(r, c + 1, vector.getY());
            return this;
        } else {
            throw new IllegalStateException("no implementation");
        }
    }

    @Override
    public MatrixDouble2D transpose() {
        MatrixDouble2D tm = new MatrixDouble2D(width(), height());
        for (int i = 0; i < height(); i++) {
            for (int j = 0; j < width(); j++) {
                tm.m[j][i] = this.m[i + rowOffset][j + colOffset];
            }
        }
        return tm;
    }

    @Override
    public MatrixDouble reset(double value) {
        for (int i = 0; i < height(); i++)
            for (int j = 0; j < width(); j++)
                m[i + rowOffset][j + colOffset] = value;
        return this;
    }

    @Override
    public <T> T unwrap(Class<T> clazz) {
        if (MatrixDouble2D.class.isAssignableFrom(clazz)) {
            return (T) this;
        }
        return null;
    }


    public static void main(String[] args) {
        double q1;
        double m1;
        MatrixDouble2D md;
        MatrixDouble span;

        md = new MatrixDouble2D(8, 8);
        md.setSubMatrix(2, 2, MatrixDouble.identity(2, -1.0));
        md.setSubMatrix(4, 2, MatrixDouble.rotation(45));
        md.setSubMatrix(6, 2, MatrixDouble.rotation(-30));
        md.dot(2.0);
        span = md.viewSpan(3, 3, 2, 2);
        span.add(MatrixDouble.identity(2, 200));

        q1 = span.getQuick(0, 0);
        m1 = md.getQuick(3, 3);

        assert m1 == q1;
    }


}
