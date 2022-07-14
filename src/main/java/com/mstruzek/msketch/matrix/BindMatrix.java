package com.mstruzek.msketch.matrix;

import Jama.Matrix;
import com.mstruzek.msketch.Point;
import com.mstruzek.sparsematrixsolver.BasicVector;

import java.util.TreeMap;

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
     * Mapa odwzorowujaca pozycje w macierzy na numer punktu w bazie Points.dbPoint
     * po kolei kazdy punkt ma po 2 wiersze
     * pierwszy integer (klucz) - id
     * drugi integer (value) - Point
     */
    TreeMap<Integer, Point> dbPoint = null; //new TreeMap<Integer,Point>()

    /**
     * Podstawowe konstruktory z wiazaniem
     */
    public BindMatrix(TreeMap<Integer, Point> db) {
        super(db.size() * 2, 1);
        this.dbPoint = db;
        System.out.println(getColumnDimension() + " - " + getRowDimension());
        this.copyFromPoints();
    }

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

    public void bind(TreeMap<Integer, Point> db) {
        this.dbPoint = db;
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
        for (Integer i : dbPoint.keySet()) {
            double x = m[k * 2][0];
            double y = m[k * 2 + 1][0];
            dbPoint.get(i).setLocation(x, y);
            k++;
        }
    }

    /**
     * Funkcja przepisuje wartosci z punktow
     * do aktualnej macierzy
     */
    public void copyFromPoints() {
        //FIXME -nie powinnismy dobierac sie do punktow za pomoca for(i++)
        // tylko wykorzystac keySet;
        double[][] m = getArray();
        int k = 0;
        for (Integer i : dbPoint.keySet()) {
            //System.out.println(i + " , " + dbPoint.get(i));
            m[k * 2][0] = dbPoint.get(i).getX();
            m[k * 2 + 1][0] = dbPoint.get(i).getY();
            k++;
        }
    }

    public static void main(String[] args) {

        Point p1 = new Point(Point.nextId(), 0.0, 0.1);
        Point p2 = new Point(Point.nextId(), 1.0, 0.2);
        Point p3 = new Point(Point.nextId(), 0.0, 0.3);
        BindMatrix bm = new BindMatrix(Point.getDbPoint());
        BindMatrix b2 = new BindMatrix(6, 1);
        b2.set(0, 0, 10);
        System.out.println(Point.getDbPoint());
        bm.set(4, 0, 10.0);
        bm.set(5, 0, 20.0);
        bm.copyToPoints();
        bm.plusEquals(b2);
        bm.copyToPoints();
        //x.plusEquals(dx); //x = x + dx
        System.out.println(bm);

        System.out.println(Point.getDbPoint());
    }

}
