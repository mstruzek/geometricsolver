package com.mstruzek.msketch.matrix;

import Jama.Matrix;
import com.mstruzek.msketch.Point;
import com.mstruzek.sparsematrixsolver.BasicVector;

/**
 * Klasa BindMatrix bedzie nam wiazala wartosci w danej macierzy z odpowiednimi
 * wartosciami w punktach
 * UWAGA BINDMATRIX NIE DZIA�A NA KOPIACH !!!!
 *
 * @author root
 */
public class BindMatrix extends Matrix {

    /**
     *
     */
    private static final long serialVersionUID = 1L;

    /**
     * Konstruktor bez wiazania specjalnie zrobiony aby dodatkowo
     * dodac toString
     *
     * @param m
     * @param n
     */
    public BindMatrix(int m, int n) {
        super(m, n, 0.0);
    }

    public BindMatrix(double[][] m) {
        super(m);
    }

    public BindMatrix(BasicVector bv) {
        super(bv.size, 1);
        double[][] m = getArray();
        for (int i = 0; i < bv.size; i++) {
            m[i][0] = bv.d[i];
        }
    }

    public void bindDbPoints() {
        //System.out.println(getColumnDimension() + " - " + getRowDimension());
        this.copyFromPoints();
    }

    @Override
    public String toString() {
        StringBuffer str = new StringBuffer();
        str.append("\nBindMatrix > Jama " + this.getRowDimension() + "x" + this.getColumnDimension() + "\n**************************************** \n");
        str.append(toString("%7.3f"));
        return str.toString();
    }

    /**
     * Ladne formatowanie jak w C
     *
     * @param format "%7.3f"
     * @return
     */
    public String toString(String format) {
        double[][] m = getArray();
        StringBuffer str = new StringBuffer();
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length - 1; j++)
                str.append(String.format(format + " ", m[i][j]));
            str.append(String.format(format, m[i][m[i].length - 1]));
            if (i < m.length - 1)
                str.append("\n");
        }
        return str.toString();
    }

    /**
     * Funkcja przepisuje wartosci z macierzy do
     * wszystkich punktow
     */
    public void copyToPoints() {
        double[][] m = getArray();
        int k = 0;
        for (Integer i : Point.dbPoint.keySet()) {
            double x = m[k * 2][0];
            double y = m[k * 2 + 1][0];
            Point.dbPoint.get(i).setLocation(x, y);
            k++;
        }
    }

    /**
     * Funkcja przepisuje wartosci z punktow
     * do aktualnej macierzy
     */
    public void copyFromPoints() {
        double[][] m = getArray();
        int k = 0;
        for (Integer i : Point.dbPoint.keySet()) {
            //System.out.println(i + " , " + dbPoint.get(i));
            m[k * 2][0] = Point.dbPoint.get(i).getX();
            m[k * 2 + 1][0] = Point.dbPoint.get(i).getY();
            k++;
        }
    }
}
