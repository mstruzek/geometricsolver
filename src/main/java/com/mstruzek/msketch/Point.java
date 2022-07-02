package com.mstruzek.msketch;

import java.util.TreeMap;


public class Point extends Vector {

    /**
     * Licznik punktow
     */
    static int counter = 0;
    /**
     * numer kolejno utworzonego punktu
     */
    int id = counter++;
    /**
     * tablica wszystkich Point�w
     */
    //static ArrayList<Point> dbPoint = new ArrayList<Point>();

    public Vector Vector = this;

    static TreeMap<Integer, Point> dbPoint = new TreeMap<Integer, Point>();

    /**
     * Konstruktor
     */
    public Point(double x, double y) {
        super(x, y);
        dbPoint.put(id, this);
    }

    public Point() {
        super(0.0, 0.0);
        dbPoint.put(id, this);
    }

    /**
     * Konstruktor kopiujacy
     */
    public Point(Vector p1) {
        super(p1.x, p1.y);
        dbPoint.put(id, this);
    }

    public String toString() {
        return "p" + id + " : " + "[ " + this.x + " , " + this.y + " ] ";

    }

    public Vector Vector() {
        return this;
    }

    public int getId() {
        return id;
    }

    public static TreeMap<Integer, Point> getDbPoint() {
        return dbPoint;
    }

    /**
     * @param args
     */
    public static void main(String[] args) {

        Point p1 = new Point(0.0, 0.1);
        Point p2 = new Point(1.0, 0.2);
        Point p3 = new Point(0.0, 0.3);
        Point p4 = new Point(p1);
        p1.setX(0.9);
        System.out.println(p1);
        System.out.println(p2);
        System.out.println(p3);
        System.out.println(p4);
        System.out.println("To pokazuje ArrayLsit :");
        System.out.println(Point.dbPoint.get(0));
        System.out.println("To pokazuje TreeMap :");
        System.out.println(Point.dbPoint.keySet().iterator());

    }

}
