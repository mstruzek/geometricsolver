package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

import java.util.Collections;
import java.util.Set;

/**
 * Klasa FreePoint - czyli wolny Punkt
 */
public class FreePoint extends GeometricObject {

    /**
     * fix control points -punkty kontrolne zafixowane
     */
    Point a = null;
    Point b = null;
    /**
     * dynamic points - punkty dynamiczne odpowiedzialne za przemiszczanie sie lini
     */
    Point p1 = null;

    /**
     * Odleglosci pomiedzy wektorami poczatkowe
     */
    double d_a_p1, d_p1_b;
    /**
     * wsp�czynnik do skalowania wzglednego dla wyznaczenia wektorow a,b
     */
    double distance = 150.0;
    /**
     * Kat rotacji wzgledem osi X ,dla wyznaczenie polozenia poczatkowego dla a,b - w stopniach
     */
    double angle = Math.toRadians(40);

    public FreePoint(Vector p) {
        this(ModelRegistry.nextPrimitiveId(), p);
    }

    public FreePoint(int id, Vector v00) {
        super(id, GeometricType.FreePoint);
        if (v00 instanceof Point) {
            p1 = (Point) v00;
            a = new Point(p1.getId() - 1, v00.minus(new Vector(distance * Math.cos(angle), distance * Math.sin(angle))));
            b = new Point(p1.getId() + 1, v00.plus(new Vector(distance * Math.cos(angle), distance * Math.sin(angle))));
        } else {
            // Punkty kontrolne wyznaczamy na podstawie distance,agnle : distance- to odleglosc wektora a i b od p1
            // Kolejnosc inicjalizacji ma znaczenie
            a = new Point(ModelRegistry.nextPointId(), v00.minus(new Vector(distance * Math.cos(angle), distance * Math.sin(angle))));
            p1 = new Point(ModelRegistry.nextPointId(), v00);
            b = new Point(ModelRegistry.nextPointId(), v00.plus(new Vector(distance * Math.cos(angle), distance * Math.sin(angle))));
            // przelicz odleglosci
            setAssociateConstraints(null);
        }
        calculateDistance();
    }

    /**
     * Funkcja oblicza dlugosci poczatkowe pomiedzy wektorami
     */
    private void calculateDistance() {
        d_a_p1 = Math.abs(p1.minus(a).length()) * 0.1;
        d_p1_b = Math.abs(b.minus(p1).length()) * 0.1;
    }

    public String toString() {
        String out = type + "*" + this.primitiveId + ": {";
        out += "a=" + a + ",";
        out += "p1=" + p1 + ",";
        out += "b=" + b + "}\n";
        return out;
    }

    @Override
    public void evaluateGuidePoints() {
        Vector va = (p1.minus(new Vector(distance * Math.cos(angle), distance * Math.sin(angle))));
        Vector vb = (p1.plus(new Vector(distance * Math.cos(angle), distance * Math.sin(angle))));
        a.setLocation(va.getX(), va.getY());
        b.setLocation(vb.getX(), vb.getY());
        // przelicz odleglosci
        calculateDistance();

        ((ConstraintFixPoint) constraints[0]).setFixVector(va);
        ((ConstraintFixPoint) constraints[1]).setFixVector(vb);

    }

    @Override
    public void evaluateForceIntensity(int row, MatrixDouble mt) {

        // 8 = 4*2 (4 punkty kontrolne)

        //F12 - sily w sprezynach
        Vector f12 = p1.minus(a).unit().product(Consts.springStiffnessLow).product(p1.minus(a).length() - d_a_p1);
        //F23
        Vector f23 = b.minus(p1).unit().product(Consts.springStiffnessLow).product(b.minus(p1).length() - d_p1_b);

        //F1 - sily na poszczegolne punkty
        mt.setVector(row + 0, 0, f12);
        //F2
        mt.setVector(row + 2, 0, f23.minus(f12));
        //F3
        mt.setVector(row + 4, 0, f23.product(-1));
    }


    @Override
    public void setStiffnessMatrix(int row, int col, MatrixDouble mt) {
        /**
         * k= I*k
         * [ -ks    ks     0;
         *    ks  -2ks   ks ;
         *     0    ks   -ks];

         */
        // K -mala sztywnosci
        MatrixDouble Ks = MatrixDouble.diagonal(Consts.springStiffnessLow, Consts.springStiffnessLow);
        MatrixDouble Km = Ks.multiplyC(-1);

        mt.plusSubMatrix(row + 0, col + 0, Km);
        mt.plusSubMatrix(row + 0, col + 2, Ks);

        mt.plusSubMatrix(row + 2, col + 0, Ks);
        mt.plusSubMatrix(row + 2, col + 2, Km.multiplyC(2.0));
        mt.plusSubMatrix(row + 2, col + 4, Ks);

        mt.plusSubMatrix(row + 4, col + 2, Ks);
        mt.plusSubMatrix(row + 4, col + 4, Km);
    }

    @Override
    public void setAssociateConstraints(Set<Integer> skipIds) {
        if (skipIds == null) skipIds = Collections.emptySet();
        ConstraintFixPoint fixPointa = new ConstraintFixPoint(ModelRegistry.nextConstraintId(skipIds), a, false);
        ConstraintFixPoint fixPointb = new ConstraintFixPoint(ModelRegistry.nextConstraintId(skipIds), b, false);
        constraints = new Constraint[2];
        constraints[0] = fixPointa;
        constraints[1] = fixPointb;
    }


    @Override
    public int getNumOfPoints() {
        return 3; //a,p1,p2
    }

    @Override
    public int getP1() {
        return p1.id;
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
    public int[] getAllPointsId() {
        int[] out = new int[3];
        out[0] = a.getId();
        out[1] = b.getId();
        out[2] = p1.getId();
        return out;
    }

    @Override
    public Point[] getAllPoints() {
        return new Point[]{a, p1, b};
    }

    @Override
    public int getA() {
        return a.id;
    }

    @Override
    public int getB() {
        return b.id;
    }

    @Override
    public int getC() {
        return -1;
    }

    @Override
    public int getD() {
        return -1;
    }

}
