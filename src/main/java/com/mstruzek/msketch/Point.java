package com.mstruzek.msketch;

import java.util.TreeMap;


public class Point extends Vector{

    public static final TreeMap<Integer,Point> dbPoint=new TreeMap<Integer,Point>();

    public static final Point EMPTY=new Point(-1, 0.0,0.0);

    /** Licznik punktow */
    public static int pointCounter =0;

    /** numer kolejno utworzonego punktu */
    protected int id;

    public Vector Vector=this;

    public Point(int id, Vector v1){
        this(id, v1.x, v1.y);
    }

    public Point(int id, double x,double y){
        super(x,y);
        this.id = id;
        if(id >=0) dbPoint.put(id,this);
    }

    public String toString(){
        return String.format("p%d : [ %7.3f , %7.3f ]", id, x, y);
    }

    public Vector Vector(){
        return this;
    }

    public int getId(){
        return id;
    }

    public static TreeMap<Integer,Point> getDbPoint(){
        return dbPoint;
    }


    public static int nextId() {
        return pointCounter++;
    }

}

