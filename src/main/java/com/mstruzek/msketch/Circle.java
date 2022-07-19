package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

import java.util.Collections;
import java.util.Set;

public class Circle extends GeometricPrimitive {

    /**
     * fix control points -punkty kontrolne zafixowane
     */
    Point a = null;
    Point b = null;
    /** dynamic points - punkty dynamiczne odpowiedzialne za przemiszczanie sie okregu*/
    /**
     * srodek okregu
     */
    Point p1 = null;
    /**
     * promien okregu
     */
    Point p2 = null;

    /**
     * Odleglosci pomiedzy wektorami poczatkowe
     */
    double d_a_p1, d_p1_p2, d_p2_b;
    /**
     * wsp�czynnik do skalowania wzglednego dla wektorow
     */
    double alfa = 2.0;

    /**
     * Wspoczynnik o ile zwiekszona jest sztywnosc glownej lini
     */
    double springAlfa = 10.0;

    /**
     * Numery wiezow  powiazane z a,b
     */
    int[] constraintsId = new int[2];

    /**
     * Konstruktor Okregu
     *
     * @param p10 srodek okregu
     * @param p20 promien
     */
    public Circle(Vector p10, Vector p20) {
        this(GeometricPrimitive.nextId(), p10, p20);
    }

    public Circle(int id, Vector v10, Vector v20) {
        super(id, GeometricPrimitiveType.Circle);

        if (v10 instanceof Point && v20 instanceof Point) {
            p1 = (Point) v10;
            p2 = (Point) v20;
            a = new Point(p1.getId() - 1, v10.sub(v20).dot(alfa).add(v10));
            b = new Point(p2.getId() + 1, v20.sub(v10).dot(alfa).add(v20));
        } else {
            //ustawienie pozycji dla punktow kontrolnych
            //Kolejnosc inicjalizacji ma znaczenie
            a = new Point(Point.nextId(), v10.sub(v20).dot(alfa).add(v10));
            p1 = new Point(Point.nextId(), v10); /// przepisujemy wartosci
            p2 = new Point(Point.nextId(), v20);
            b = new Point(Point.nextId(), v20.sub(v10).dot(alfa).add(v20));
            setAssociateConstraints(null);
        }
        calculateDistance();
    }

    public String toString() {
        return this.type + "*" +
            this.primitiveId + ": {" +
            ",a=" + a +
            ",p1=" + p1 +
            ",p2=" + p2 +
            ",b=" + b +
            "}\n";
    }

    /**
     * Funkcja oblicza dlugosci poczatkowe pomiedzy wektorami
     */
    private void calculateDistance() {
        d_a_p1 = Math.abs(p1.sub(a).length());
        d_p1_p2 = Math.abs(p2.sub(p1).length());
        d_p2_b = Math.abs(b.sub(p2).length());
    }

    @Override
    public void evaluateGuidePoints() {
        Vector va = (Vector) (p1.sub(p2).dot(alfa).add(p1));
        Vector vb = (Vector) (p2.sub(p1).dot(alfa).add(p2));
        a.setLocation(va.getX(), va.getY());
        b.setLocation(vb.getX(), vb.getY());

        calculateDistance();

        //uaktulaniamy wiezy
        ((ConstraintFixPoint) Constraint.dbConstraint.get(constraints[0])).setFixVector(va);
        ((ConstraintFixPoint) Constraint.dbConstraint.get(constraints[1])).setFixVector(vb);
    }

    @Override
    public void setForce(int row, MatrixDouble force) {
        // 8 = 4*2 (4 punkty kontrolne)
        Vector f12 = p1.sub(a).unit().dot(Consts.springStiffnessLow).dot(p1.sub(a).length() - d_a_p1);                      //F12 - sily w sprezynach
        Vector f23 = p2.sub(p1).unit().dot(Consts.springStiffnessHigh * springAlfa).dot(p2.sub(p1).length() - d_p1_p2);     //F23
        Vector f34 = b.sub(p2).unit().dot(Consts.springStiffnessLow).dot(b.sub(p2).length() - d_p2_b);                      //F34

        //F1 - silu na poszczegolne punkty
        force.setVector(row + 0, 0, f12);
        //F2
        force.setVector(row + 2, 0, f23.sub(f12));
        //F3
        force.setVector(row + 4, 0, f34.sub(f23));
        //F4
        force.setVector(row + 6, 0, f34.dot(-1.0));
    }

    @Override
    public void setJacobian(int row, int col, MatrixDouble mt) {
        // a ,p1 ,p2 ,b = 4*2 = 8x8
        /**
         * k= I*k
         * [ -ks    ks      0  	  0;
         *    ks  -ks-kb   kb  	  0;
         *     0    kb   -ks-kb   ks;
         *     0  	 0     ks    -ks];
         */
        // K -mala sztywnosci
        MatrixDouble Ks = MatrixDouble.diagonal(Consts.springStiffnessLow, Consts.springStiffnessLow);
        // K - duza szytwnosci
        MatrixDouble Kb = MatrixDouble.diagonal(Consts.springStiffnessHigh * springAlfa, Consts.springStiffnessHigh * springAlfa);
        // -Ks-Kb
        MatrixDouble Ksb = Ks.dotC(-1).add(Kb.dotC(-1));

        mt.addSubMatrix(row + 0, col + 0, Ks.dotC(-1));
        mt.addSubMatrix(row + 0, col + 2, Ks);
        mt.addSubMatrix(row + 2, col + 0, Ks);
        mt.addSubMatrix(row + 2, col + 2, Ksb);
        mt.addSubMatrix(row + 2, col + 4, Kb);
        mt.addSubMatrix(row + 4, col + 2, Kb);
        mt.addSubMatrix(row + 4, col + 4, Ksb);
        mt.addSubMatrix(row + 4, col + 6, Ks);
        mt.addSubMatrix(row + 6, col + 4, Ks);
        mt.addSubMatrix(row + 6, col + 6, Ks.dotC(-1));
    }

    @Override
    public void setAssociateConstraints(Set<Integer> skipIds) {
        if (skipIds == null) skipIds = Collections.emptySet();
        ConstraintFixPoint fixPointa = new ConstraintFixPoint(Constraint.nextId(skipIds), a, false);
        ConstraintFixPoint fixPointb = new ConstraintFixPoint(Constraint.nextId(skipIds), b, false);
        constraints = new int[2];
        constraints[0] = fixPointa.constraintId;
        constraints[1] = fixPointb.constraintId;
    }

    @Override
    public int getNumOfPoints() {
        //a,b,p1,p2
        return 4;
    }

    @Override
    public int getP1() {
        return p1.id;
    }

    @Override
    public int getP2() {
        return p2.id;
    }

    @Override
    public int getP3() {
        return -1;
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

    @Override
    public int[] getAllPointsId() {
        int[] out = new int[4];
        out[0] = a.getId();
        out[1] = b.getId();
        out[2] = p1.getId();
        out[3] = p2.getId();
        return out;
    }

}
