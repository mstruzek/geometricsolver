package com.mstruzek.msketch;

import java.util.TreeMap;


public class Point extends Vector{

    public static final TreeMap<Integer,Point> dbPoint=new TreeMap<Integer,Point>();

    public static final Point EMPTY=new Point(-1);

    /**
     * Licznik punktow
     */
    static int counter=0;
    /**
     * numer kolejno utworzonego punktu
     */
    protected int id;
    /**
     * tablica wszystkich Point�w
     */
    //static ArrayList<Point> dbPoint = new ArrayList<Point>();

    public Vector Vector=this;

    public Point(){
        this(nextId(),0.0,0.0);
    }

    public Point(int id){
        this(id, 0.0,0.0);
    }

    public Point(int id, Vector v1){
        this(id, v1.x, v1.y);
    }

    public Point(Vector p1){
        this(nextId(), p1.x,p1.y);
    }

    public Point(double x,double y){
        this(nextId(), x, y);
    }

    public Point(int id, double x,double y){
        super(x,y);
        this.id = id;
        dbPoint.put(id,this);
    }

    public String toString(){
        return "p"+id+" : "+"[ "+this.x+" , "+this.y+" ] ";

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
        return counter++;
    }
}

