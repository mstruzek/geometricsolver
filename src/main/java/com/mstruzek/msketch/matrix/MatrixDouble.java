package com.mstruzek.msketch.matrix;

import com.mstruzek.msketch.Vector;

/**
 * Klasa reprezentuje macierz
 * + dodatkowe operacje
 *
 * @author root
 */
public class MatrixDouble {

    /**
     * zmienna przechowujaca nasze elementy
     */
    public double[][] m = null;
    /**
     * szerokosc -ilosc kolumn
     */
    int columns;
    /**
     * wysokosc - ilosc wiersze
     */
    int rows;


    /**
     * Konstruktor macierzy
     *
     * @param columns szerokosc -ilosc kolumn
     * @param rows    wysokosc - ilosc wiersze
     */
    public MatrixDouble(int rows, int columns) {
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
    public MatrixDouble(Vector vec, boolean columnType) {
        if(columnType) {
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

    public void clear() {
        //FIXME - to poprawic bo pozniej to bedzie brakowac pamieci
        this.m = null;
        this.rows = 0;
        this.columns = 0;
    }

    /**
     * Get element of matrix
     *
     * @param i - row
     * @param j - column
     * @return double value
     */
    public double get(int i, int j) {
        return this.m[i][j];
    }


    /**
     * Number of columns
     *
     * @return
     */
    public int getWidth() {
        return columns;
    }

    /**
     * Number of rows
     *
     * @return
     */
    public int getHeight() {
        return rows;
    }

    /**
     * Dodaj skalar do kazdej pozycji macierzy
     *
     * @param a
     * @return aktualan macierz this
     */
    public MatrixDouble add(double a) {
        for(int i = 0; i < this.m.length; i++) {
            for(int j = 0; j < this.m[i].length; j++) {
                this.m[i][j] = this.m[i][j] + a;
            }
        }
        return this;
    }


    /**
     * Adds a matrix to another matrix, which should be of the same dimension.
     *
     * @returns The resulting matrix , actual matrix this
     */
    public MatrixDouble add(MatrixDouble rhs) {
        assertEqualDimensions(this, rhs);
        for(int i = 0; i < this.m.length; i++) {
            for(int j = 0; j < this.m[i].length; j++) {
                this.m[i][j] = this.m[i][j] + rhs.m[i][j];
            }
        }
        return this;
    }

    private static void assertEqualDimensions(MatrixDouble left, MatrixDouble right) {
        if(left.rows != right.rows || left.columns != right.columns) {
            throw new Error("Matrices must be of the same dimension");
        }
    }

    /**
     * Adds a matrix to another matrix, which should be of the same dimension.
     *
     * @returns copy of this matrix
     */
    public MatrixDouble addC(MatrixDouble rhs) {
        assertEqualDimensions(this, rhs);
        MatrixDouble rm = new MatrixDouble(this.rows, this.columns);
        for(int i = 0; i < this.m.length; i++) {
            for(int j = 0; j < this.m[i].length; j++) {
                rm.m[i][j] = this.m[i][j] + rhs.m[i][j];
            }
        }
        return rm;
    }

    /**
     * Mnozezenie kazdego elementu macierzy przez skalar
     *
     * @param c skalar
     * @return this
     */
    public MatrixDouble dot(double c) {
        for(int i = 0; i < getHeight(); i++) {
            for(int j = 0; j < getWidth(); j++) {
                this.m[i][j] *= c;
            }
        }
        return this;
    }

    /**
     * Mnozezenie kazdego elementu macierzy przez skalar
     * zwracana jest kopia , aktualna macierz niezmieniona
     *
     * @param c skalar
     * @return kopia macierzy aktualnej
     */
    public MatrixDouble dotC(double c) {
        MatrixDouble mt = this.copy();
        for(int i = 0; i < getHeight(); i++) {
            for(int j = 0; j < getWidth(); j++) {
                mt.m[i][j] *= c;
            }
        }
        return mt;
    }

    /**
     * Generate a copy of a matrix
     *
     * @return A copy of this matrix
     */
    public MatrixDouble copy() {
        MatrixDouble array = new MatrixDouble(this.m.length, this.m[0].length);
        for(int i = 0; i < array.m.length; i++)
            System.arraycopy(this.m[i], 0, array.m[i], 0, this.m[i].length);
        return array;
    }


    /**
     * Funkcja wstawia macierz mt na dana pozycje w akutalnej macierzy
     *
     * @param firstRow    poczatkowy wiersz
     * @param firstColumn poczatkowa kolumna
     * @param mt          macierz do wstawienia
     * @return this
     */
    public MatrixDouble setSubMatrix(int firstRow, int firstColumn, MatrixDouble mt) {
        //sprawdzamy czy macierz wstawiana nie jest za duza
        if(this.getHeight() >= (firstRow + mt.getHeight())) {
            if(this.getWidth() >= (firstColumn + mt.getWidth())) {
                //mozna wstawic
                //System.arraycopy(this.m[i + i1], j1, array.m[i], 0, j2 - j1 + 1);
                for(int k = 0; k < mt.getHeight(); k++) {
                    System.arraycopy(mt.m[k], 0, this.m[k + firstRow], firstColumn, mt.m[k].length);
                }
                return this;
            } else {
                return null;
            }
        } else {
            return null;
        }
    }


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
    public MatrixDouble addSubMatrix(int firstRow, int firstColumn, MatrixDouble mt) {
        if(this.getHeight() < (firstRow + mt.getHeight()) || this.getWidth() < (firstColumn + mt.getWidth())) {
            throw new Error("matrix dimension out of bounds");
        }

        for(int i = 0; i < mt.getHeight(); i++) {
            for(int j = 0; j < mt.getWidth(); j++) {
                m[i + firstRow][j + firstColumn] += mt.m[i][j];
            }
        }
        return this;
    }


    /**
     * Transposes an mxn matrix into an nxm matrix. Each row of the input matrix becomes a column in the
     * output matrix.
     *
     * @return transposed cloned matrix.
     */
    public MatrixDouble transpose() {
        MatrixDouble tm = new MatrixDouble(columns, rows);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                tm.m[j][i] = this.m[i][j];
            }
        }
        return tm;
    }

    /**
     * Transposes an mxn matrix into an nxm matrix. Each row of the input matrix becomes a column in the
     * output matrix.
     *
     * @return copy , transposed matrix .
     */
    public MatrixDouble transposeC() {
        MatrixDouble tM = new MatrixDouble(this.m[0].length, this.m.length);
        for(int i = 0; i < tM.m.length; i++){
            for(int j = 0; j < tM.m[0].length; j++){
                tM.m[i][j] = this.m[j][i];
            }
        }
        return tM;
    }


    /**
     * Access the internal two-dimensional array.
     *
     * @return Pointer to the two-dimensional array of matrix elements.
     */
    public double[][] getArray() {
        return this.m;
    }


    /**
     * Funkcja tworzy nowa macierz na podstawie tablicy double[][]
     *
     * @param mat macierz prostokatna
     */
    public static MatrixDouble createFromArray(double[][] mat) {
        MatrixDouble mt = new MatrixDouble(mat.length, mat[0].length);
        for(int i = 0; i < mat.length; i++) {
            System.arraycopy(mat[i], 0, mt.m[i], 0, mat[i].length);
        }
        return mt;
    }

    /**
     * Generates an m x m identity matrix. Result has ones along the diagonal
     * and zeros everywhere else. Example:<br>
     *
     * @param m an integer > 0.
     * @return m x m two dimensional MatrixDouble.
     */
    public static MatrixDouble identity(int m) {
        return diagonal(m, 1.0);
    }

    /**
     * Returns an m x m matrix. result has constants along the diagonal
     * and zeros everywhere else. Example:<br>
     *
     * @param m an integer > 0.
     * @param c Constant that lies along diagonal. Set c=1 for identity matrix.
     * @return m x m 2D array of doubles.
     */
    public static MatrixDouble diagonal(int m, double c) {
        if(m < 1)
            throw new IllegalArgumentException("First argument must be > 0");
        MatrixDouble I = new MatrixDouble(m, m);
        for(int i = 0; i < I.m.length; i++)
            I.m[i][i] = c;
        return I;
    }

    /**
     * Returns an m x m matrix. result has specified values along the diagonal
     * and zeros everywhere else. Example:<br>
     *
     * @param c Values that lies along diagonal.
     * @return c.length x c.length 2D array of doubles.
     */
    public static MatrixDouble diagonal(double... c) {
        MatrixDouble I = new MatrixDouble(c.length, c.length);
        for(int i = 0; i < I.m.length; i++)
            I.m[i][i] = c[i];
        return I;
    }

    /**
     * Zwraca macierz rotacji obrocona o kat alfa
     *
     * @param alfa
     * @return
     */
    public static MatrixDouble getRotation2x2(double alfa) {
        MatrixDouble r = new MatrixDouble(2, 2);

        r.m[0][0] = Math.cos(Math.toRadians(alfa));
        r.m[0][1] = Math.sin(Math.toRadians(-alfa));
        r.m[1][0] = Math.sin(Math.toRadians(alfa));
        r.m[1][1] = Math.cos(Math.toRadians(alfa));
        return r;

    }

    /**
     * Fills an m x n matrix of doubles with constant c. Example:<br>
     *
     * @param m Number of rows in matrix
     * @param n Number of columns in matrix
     * @param c constant that fills matrix
     * @return m x n 2D array of constants c
     */
    public static MatrixDouble fill(int m, int n, double c) {
        MatrixDouble o = new MatrixDouble(m, n);
        for(int i = 0; i < o.m.length; i++)
            for(int j = 0; j < o.m[i].length; j++)
                o.m[i][j] = c;
        return o;
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
    public static MatrixDouble mergeByColumn(MatrixDouble... MD) {
        int maxRows = 0;
        int maxColumns = 0;
        for(int i = 0; i < MD.length; i++) {
            maxRows += MD[i].getHeight();
            if(MD[i].getWidth() > maxColumns) maxColumns = MD[i].getWidth();
        }
        MatrixDouble MT = new MatrixDouble(maxRows, maxColumns);
        int currentRow = 0;
        for(int i = 0; i < MD.length; i++) {
            for(int j = 0; j < MD[i].getHeight(); j++) { //j - numer wiersza w danej macierzy
                System.arraycopy(MD[i].m[j], 0, MT.m[currentRow + j], 0, MD[i].m[j].length);
            }
            currentRow += MD[i].getHeight();
        }
        return MT;
    }

    /**
     * Dodaje do siebie Macierze w ten sposub ze kazda kolejna macierz
     * jest obok drugiej w wierszu , macierze powinny miec ta sama ilosc wierszy ale nie koniecznie
     * <p>
     * M = mergeByRow(A,B,C);
     * M= [ A,B,C];
     *
     * @param MD - lista macierzy do polaczenia
     * @return
     */
    public static MatrixDouble mergeByRow(MatrixDouble... MD) {
        int maxRows = 0;
        int maxColumns = 0;
        for(int i = 0; i < MD.length; i++) {
            maxColumns += MD[i].getWidth();
            if(MD[i].getHeight() > maxRows) maxRows = MD[i].getHeight();
        }
        MatrixDouble MT = new MatrixDouble(maxRows, maxColumns);
        int currentColumn = 0;
        for(int i = 0; i < MD.length; i++) {
            for(int j = 0; j < MD[i].getHeight(); j++) { //j - numer wiersza w danej macierzy
                System.arraycopy(MD[i].m[j], 0, MT.m[j], currentColumn, MD[i].m[j].length);
            }
            currentColumn += MD[i].getWidth();
        }
        return MT;
    }

    /**
     * Generates a string that holds a nicely organized version of a matrix or array.
     * Uses format specifier, e.g. "%5.3f", "%11.1E", ...
     * An extra space is automatically included.
     * Note the lower case S. It's tostring() not toString().
     * Example:<br>
     * <code>
     * double[][] a = random(2, 3);<br>
     * System.out.println(tostring("%7.3f", a));<br>
     * result is:<br>
     * 0.654   0.115   0.422<br>
     * 0.560   0.839   0.280<br>
     * </code>
     *
     * @return A string of a nicely organized version of the matrix or array.
     */
    public String toString(String format) {
        StringBuffer str = new StringBuffer();
        for(int i = 0; i < this.m.length; i++) {
            for(int j = 0; j < this.m[i].length - 1; j++)
                str.append(String.format(format + " ", this.m[i][j]));
            str.append(String.format(format, this.m[i][this.m[i].length - 1]));
            if(i < this.m.length - 1)
                str.append("\n");
        }
        return str.toString();
    }

    /**
     * Generates a string that holds a nicely organized space-seperated version of a matrix or array.
     * An extra space is automatically included.
     * Note the lower case S. It's tostring() not toString().
     *
     * @return A string of a nicely organized version of the matrix or array.
     */
    public String toString() {
        StringBuffer str = new StringBuffer();
        str.append("\nMatrixDouble.m " + this.getHeight() + "x" + this.getWidth() + "\n**************************************** \n");
        str.append(toString("%7.3f"));
        return str.toString();
    }

    /**
     * @param args
     */
    public static void main(String[] args) {
        MatrixDouble m1 = MatrixDouble.fill(1, 3, 1.0);
        System.out.println(m1);
        //m1.resize(3, 3);
        MatrixDouble m2 = MatrixDouble.fill(1, 2, 5.0);
        //m1.resize(2, 2);
        System.out.println(m2);
        System.out.println(MatrixDouble.mergeByRow(m1, m2));
        //MatrixDouble m3 = m1.addSubMatrixC(0, 0, m2);
        //System.out.println(m3);
        //System.out.println(m1);
        double[][] tab = {{1, 2, 3}, {4, 5, 6}};
        MatrixDouble mg = MatrixDouble.createFromArray(tab);
        System.out.println(mg.transpose());
    }

}
