package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

import java.util.Collections;
import java.util.Set;


/// FIX Point recalculated form P1 and P2, from Persistence and Serialization!

//FIXME - trzeba zaburzac punkty jezeli sa singularity 

/**
 * Klasa Line - model linii w szkicowniku
 * pomiedzy punktami [a,p1] i [p2,b] -zastosowane sprezyny slabe - springStiffnessLow
 * pomiedzy pynktami [p1,p2] - sprezyny mocne springStiffnessHigh
 */
public class Line extends GeometricObject {

    /**
     * fix control points -punkty kontrolne zafixowane
     */
    Point a = null;
    Point b = null;
    /**
     * dynamic points - punkty dynamiczne odpowiedzialne za przemiszczanie sie lini
     */
    Point p1 = null;
    Point p2 = null;

    /**
     * Odleglosci pomiedzy wektorami poczatkowe
     */
    double d_a_p1, d_p1_p2, d_p2_b;

    /**
     * wspolczynnik do skalowania wzglednego dla wektorow
     */
    double alfa = 0.5;

    /**
     * Numery wiezow  powiazane z a,b
     */
    int[] constraintsId = new int[2];

    public Line(Vector p10, Vector p20) {
        this(ModelRegistry.nextPrimitiveId(), p10, p20);
    }

    public Line(int id, Vector v10, Vector v20) {
        super(id, GeometricType.Line);

        if (v10 instanceof Point && v20 instanceof Point) {
            p1 = (Point) v10;
            p2 = (Point) v20;
            a = new Point(p1.getId() - 1, v10.minus(v20).product(alfa).plus(v10));
            b = new Point(p2.getId() + 1, v20.minus(v10).product(alfa).plus(v20));
        } else {
            //ustawienie pozycji dla punktow kontrolnych
            //Kolejnosc inicjalizacji ma znaczenie
            a = new Point(ModelRegistry.nextPointId(), v10.minus(v20).product(alfa).plus(v10));
            p1 = new Point(ModelRegistry.nextPointId(), v10);//przepisujemy wartosci
            p2 = new Point(ModelRegistry.nextPointId(), v20);
            b = new Point(ModelRegistry.nextPointId(), v20.minus(v10).product(alfa).plus(v20));
            setAssociateConstraints(null);
        }
        //FIXME  -Trzeba pomyslec o naciagu wstepnym sprezyn
        calculateDistance();
        //ustawiamy wiezy dla punktow a i b , przechowujemy lokalnie numery wiezow
    }

    public String toString() {
        String out = type + "" + "*" + this.primitiveId + ": {";
        out += "a=" + a + ",";
        out += "p1=" + p1 + ",";
        out += "p2=" + p2 + ",";
        out += "b=" + b + "}\n";
        return out;
    }

    /**
     * Funkcja oblicza dlugosci poczatkowe pomiedzy wektorami
     */
    private void calculateDistance() {

        d_a_p1 = Math.abs(p1.minus(a).length());
        d_p1_p2 = Math.abs(p2.minus(p1).length());
        d_p2_b = Math.abs(b.minus(p2).length());
    }

    @Override
    public void evaluateGuidePoints() {

        Vector va = (Vector) (p1.minus(p2).product(alfa).plus(p1));
        Vector vb = (Vector) (p2.minus(p1).product(alfa).plus(p2));
        a.setLocation(va.getX(), va.getY());
        b.setLocation(vb.getX(), vb.getY());

        calculateDistance();

        //uaktulaniamy wiezy
        ((ConstraintFixPoint) constraints[0]).setFixVector(va);
        ((ConstraintFixPoint) constraints[1]).setFixVector(vb);
    }

    @Override
    public void evaluateForceIntensity(int row, MatrixDouble mt) {
        // 8 = 4*2 (4 punkty kontrolne)
        Vector f12 = p1.minus(a).unit().product(Consts.springStiffnessLow).product(p1.minus(a).length() - d_a_p1);          //F12 - sily w sprezynach
        Vector f23 = p2.minus(p1).unit().product(Consts.springStiffnessHigh).product(p2.minus(p1).length() - d_p1_p2);      //F23
        Vector f34 = b.minus(p2).unit().product(Consts.springStiffnessLow).product(b.minus(p2).length() - d_p2_b);          //F34

        //F1 - silu na poszczegolne punkty
        mt.setVector(row + 0, 0, f12);
        //F2
        mt.setVector(row + 2, 0, f23.minus(f12));
        //F3
        mt.setVector(row + 4, 0, f34.minus(f23));
        //F4
        mt.setVector(row + 6, 0, f34.product(-1.0));

    }

    @Override
    public void setStiffnessMatrix(int row, int col, MatrixDouble mt) {

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
        MatrixDouble Kb = MatrixDouble.diagonal(Consts.springStiffnessHigh, Consts.springStiffnessHigh);
        // -Ks-Kb
        MatrixDouble Ksb = Ks.multiplyC(-1).plus(Kb.multiplyC(-1));

        //wiersz pierwszy
        mt.plusSubMatrix(row + 0, col + 0, Ks.multiplyC(-1));
        mt.plusSubMatrix(row + 0, col + 2, Ks);
        mt.plusSubMatrix(row + 2, col + 0, Ks);
        mt.plusSubMatrix(row + 2, col + 2, Ksb);
        mt.plusSubMatrix(row + 2, col + 4, Kb);
        mt.plusSubMatrix(row + 4, col + 2, Kb);
        mt.plusSubMatrix(row + 4, col + 4, Ksb);
        mt.plusSubMatrix(row + 4, col + 6, Ks);
        mt.plusSubMatrix(row + 6, col + 4, Ks);
        mt.plusSubMatrix(row + 6, col + 6, Ks.multiplyC(-1));
    }

    @Override
    public void setAssociateConstraints(Set<Integer> skipIds) {
        if (skipIds == null) skipIds = Collections.emptySet();
        ConstraintFixPoint fixeda = new ConstraintFixPoint(ModelRegistry.nextConstraintId(skipIds), a, false);
        ConstraintFixPoint fixedb = new ConstraintFixPoint(ModelRegistry.nextConstraintId(skipIds), b, false);
        constraints = new Constraint[2];
        constraints[0] = fixeda;
        constraints[1] = fixedb;
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
    public int[] getAllPointsId() {
        int[] out = new int[4];
        out[0] = a.getId();
        out[1] = b.getId();
        out[2] = p1.getId();
        out[3] = p2.getId();
        return out;
    }

    @Override
    public Point[] getAllPoints() {
        return new Point[]{a, p1, p2, b};
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
