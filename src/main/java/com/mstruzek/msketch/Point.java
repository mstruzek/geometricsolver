package com.mstruzek.msketch;


public class Point extends Vector {

    public static final Point EMPTY = new Point(-1, 0.0, 0.0);

    /**
     * numer kolejno utworzonego punktu
     */
    protected int id;


    public Point(int id, Vector v1) {
        this(id, v1.getX(), v1.getY());
    }

    public Point(int id, double x, double y) {
        super(x, y);
        this.id = id;
    }

    public String toString() {
        return String.format("p%d : [ %7.3f , %7.3f ]", id, getX(), getY());
    }

    public Vector Vector() {
        return this;
    }

    public int getId() {
        return id;
    }


}

