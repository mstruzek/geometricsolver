package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

import java.util.Set;

/**
 * Klasa reprezentujaca zafiksowana linie,
 * zawiera wersory osi X,Y;
 */
public class FixLine extends GeometricPrimitive {

    /**
     * fix control points
     */
    Vector a = null;
    Vector b = null;

    //Pierwotne Linie naszego szkicownika
    static FixLine X = new FixLine(-1, new Vector(0.0, 0.0), new Vector(100.0, 0.0));
    static FixLine Y = new FixLine(-2, new Vector(0.0, 0.0), new Vector(0.0, 100.0));

    public FixLine(int id, Vector a1, Vector b1) {
        super(id, GeometricPrimitiveType.FixLine);
        a = new Vector(a1.getX(), a1.getY());
        b = new Vector(b1.getX(), b1.getY());
    }

    public String toString() {
        String out = type + "*" + this.primitiveId + ": {";
        out += "a = " + a + ",";
        out += "b = " + b + "}\n";
        return out;
    }

    @Override
    public void evaluateGuidePoints() {

    }

    @Override
    public void setForce(int row, MatrixDouble dest) {

    }


    @Override
    public void setJacobian(int row, int col, MatrixDouble dest) {
        //poniewaz nie ma punktow kontrolnych to brak macierzy
    }

    @Override
    public void setAssociateConstraints(Set<Integer> skipIds) {
        //brak punktow kontrolnych
    }


    @Override
    public int getNumOfPoints() {
        return 0; //brak pointow
    }

    @Override
    public int getP1() {
        return -1;
    }

    @Override
    public int getP2() {
        return -1;
    }

    @Override
    public int getP3() {
        return -1;
    }

    @Override
    public int getA() {
        return -1;
    }

    @Override
    public int getB() {
        return -1;
    }

    @Override
    public int getC() {
        return -1;
    }

    @Override
    public int getD() {
        return -1;
    }

    @Override
    public int[] getAllPointsId() {
        return null;
    }
}
