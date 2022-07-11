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
public class Line extends GeometricPrimitive {

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
        this(GeometricPrimitive.nextId(), p10, p20);
    }

    public Line(int id, Vector v10, Vector v20) {
        super(id, GeometricPrimitiveType.Line);

        if (v10 instanceof Point && v20 instanceof Point) {
            p1 = (Point) v10;
            p2 = (Point) v20;
            a = new Point(p1.getId() - 1, v10.sub(v20).dot(alfa).add(v10));
            b = new Point(p2.getId() + 1, v20.sub(v10).dot(alfa).add(v20));
        } else {
            //ustawienie pozycji dla punktow kontrolnych
            //Kolejnosc inicjalizacji ma znaczenie
            a = new Point(Point.nextId(), v10.sub(v20).dot(alfa).add(v10).x, v10.sub(v20).dot(alfa).add(v10).y);
            p1 = new Point(Point.nextId(), v10.x, v10.y);//przepisujemy wartosci
            p2 = new Point(Point.nextId(), v20.x, v20.y);
            b = new Point(Point.nextId(), v20.sub(v10).dot(alfa).add(v20).x, v20.sub(v10).dot(alfa).add(v20).y);
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

        d_a_p1 = Math.abs(p1.sub(a).length());
        d_p1_p2 = Math.abs(p2.sub(p1).length());
        d_p2_b = Math.abs(b.sub(p2).length());
    }

    @Override
    public void recalculateControlPoints() {

        Vector va = (Vector) (p1.sub(p2).dot(alfa).add(p1));
        Vector vb = (Vector) (p2.sub(p1).dot(alfa).add(p2));
        a.setLocation(va.x, va.y);
        b.setLocation(vb.x, vb.y);

        calculateDistance();

        //uaktulaniamy wiezy
        ((ConstraintFixPoint) Constraint.dbConstraint.get(constraints[0])).setFixVector(va);
        ((ConstraintFixPoint) Constraint.dbConstraint.get(constraints[1])).setFixVector(vb);
    }

    @Override
    public void setForce(int r, MatrixDouble mt) {
        // 8 = 4*2 (4 punkty kontrolne)
        Vector f12 = p1.sub(a).unit().dot(Consts.springStiffnessLow).dot(p1.sub(a).length() - d_a_p1);          //F12 - sily w sprezynach
        Vector f23 = p2.sub(p1).unit().dot(Consts.springStiffnessHigh).dot(p2.sub(p1).length() - d_p1_p2);      //F23
        Vector f34 = b.sub(p2).unit().dot(Consts.springStiffnessLow).dot(b.sub(p2).length() - d_p2_b);          //F34

        mt.setVector(r + 0, 0, f12);                     //F1 - silu na poszczegolne punkty
        mt.setVector(r + 2, 0, f23.sub(f12));            //F2
        mt.setVector(r + 4, 0, f34.sub(f23));            //F3
        mt.setVector(r + 6, 0, f34.dot(-1.0));           //F4
    }


    @Override
    public void setJacobian(int r, int c, MatrixDouble mt) {

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
        MatrixDouble Ksb = Ks.dotC(-1).addSubMatrix(0, 0, Kb.dotC(-1));

        //wiersz pierwszy
        mt.addSubMatrix(r + 0, c + 0, Ks.dotC(-1));
        mt.addSubMatrix(r + 0, c + 2, Ks);
        mt.addSubMatrix(r + 2, c + 0, Ks);
        mt.addSubMatrix(r + 2, c + 2, Ksb);
        mt.addSubMatrix(r + 2, c + 4, Kb);
        mt.addSubMatrix(r + 4, c + 2, Kb);
        mt.addSubMatrix(r + 4, c + 4, Ksb);
        mt.addSubMatrix(r + 4, c + 6, Ks);
        mt.addSubMatrix(r + 6, c + 4, Ks);
        mt.addSubMatrix(r + 6, c + 6, Ks.dotC(-1));
    }

    @Override
    public void setAssociateConstraints(Set<Integer> skipIds) {
        if (skipIds == null) skipIds = Collections.emptySet();
        ConstraintFixPoint fixeda = new ConstraintFixPoint(Constraint.nextId(skipIds), a, false);
        ConstraintFixPoint fixedb = new ConstraintFixPoint(Constraint.nextId(skipIds), b, false);
        constraints = new int[2];
        constraints[0] = fixeda.constraintId;
        constraints[1] = fixedb.constraintId;
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
