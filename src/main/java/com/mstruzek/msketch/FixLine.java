package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

/**
 * Klasa reprezentujaca zafiksowana linie,
 * zawiera wersory osi X,Y;
 */
public class FixLine extends GeometricPrimitive{

    /**
     * fix control points
     */
    Vector a=null;
    Vector b=null;

    //Pierwotne Linie naszego szkicownika
    static FixLine X=new FixLine(-1, new Vector(0.0,0.0),new Vector(100.0,0.0));
    static FixLine Y=new FixLine(-2, new Vector(0.0,0.0),new Vector(0.0,100.0));

    public FixLine(int id, Vector a1,Vector b1){
        super(id, GeometricPrimitiveType.FixLine);
        a=new Vector(a1);
        b=new Vector(b1);
    }

    public String toString(){
        String out=type+"*"+this.primitiveId+": {";
        out+="a = "+a+",";
        out+="b = "+b+"}\n";
        return out;
    }

    @Override
    public void recalculateControlPoints(){

    }

    @Override
    public MatrixDouble getForceJacobian(){
        //poniewaz nie ma punktow kontrolnych to brak macierzy
        return null;
    }

    @Override
    public int[] setAssociateConstraints(){
        //brak punktow kontrolnych
        return null;
    }

    @Override
    public MatrixDouble getForce(){
        //nic nie ma - brak sprezyn
        return null;
    }

    @Override
    public int getNumOfPoints(){
        return 0; //brak pointow
    }

    @Override
    public int getP1(){
        return -1;
    }

    @Override
    public int getP2(){
        return -1;
    }

    @Override
    public int getP3(){
        return -1;
    }

    @Override
    public int getA(){
        return -1;
    }

    @Override
    public int getB(){
        return -1;
    }

    @Override
    public int getC(){
        return -1;
    }

    @Override
    public int getD(){
        return -1;
    }

    @Override
    public int[] getAllPointsId(){
        return null;
    }
}
