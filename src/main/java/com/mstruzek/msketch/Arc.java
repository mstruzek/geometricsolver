package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

import java.util.Collections;
import java.util.Set;

/**
 * Luk geometryczny
 * - sklada sie z srodka okregu i dwoch punktow krancowych luku
 * W naszym wypatku to konstrukcja zlozona z linii (czyli 2 punkty na luku)
 * + FreePoint -srodek okregu + wiez rownej dlugosci pomiedzy punktami a srodkiem
 */
//FIXME - UWAGA przy wstawianiu LUKU nalezy pamietac aby promien znajdowal sie na symetralnej pomiedzy punktami p2,p3
public class Arc extends GeometricPrimitive {

    /**
     * fix control points -punkty kontrolne zafixowane
     */
    Point a = null;
    Point b = null;
    Point c = null;
    Point d = null;
    /**
     * dynamic points - punkty dynamiczne odpowiedzialne za przemiszczanie sie lini
     */
    Point p1 = null;
    Point p2 = null;
    Point p3 = null;

    // FreePoint(a,p1,b) , Line(c,p2,p3,d)

    // FIXME -jakis blad na jakobianie i silach nie zbiega sie :(
    /** KOLEJNOSC INICJALIZACJI MA ZNACZENIE - PATRZ KONSTRUKTOR */

    /**
     * Odleglosci pomiedzy wektorami poczatkowe
     */
    double d_a_p1, d_b_p1, d_p1_p2, d_p1_p3, d_p3_d, d_p2_c;
    /**
     * wspolczynnik do skalowania wzglednego dla wektorow
     */
    double alfa = 1.5;

    /**
     * Luk
     *
     * @param v10 srodek okregu
     * @param v20 pierwszy koniec luku
     */
    public Arc(Vector v10, Vector v20, Vector v30) {
        this(GeometricPrimitive.nextId(), v10, v20, v30);
    }

    public Arc(int id, Vector v10, Vector v20, Vector v30) {
        super(id, GeometricPrimitiveType.Arc);

        // FreePoint(a,p1,b)
        Vector va, vb, vc, vd, v3;


        if (v10 instanceof Point && v20 instanceof Point) {

            p1 = (Point) v10;
            p2 = (Point) v20;
            p3 = (Point) v30;

            va = v10.add(v10.sub(v20).dot(alfa));
            vb = v10.add(v20.sub(v10).Rot(90).dot(alfa));
            vc = v20.add(v20.sub(v10).dot(alfa));
            vd = v30.add(v20.sub(v10).Rot(-90).dot(alfa));

            a = new Point(p1.getId() - 4, va.x, va.y);
            b = new Point(p1.getId() - 3, vb.x, vb.y);
            c = new Point(p1.getId() - 2, vc.x, vc.y);
            d = new Point(p1.getId() - 1, vd.x, vd.y);

        } else {
            v3 = v10.add(v20.sub(v10).Rot(-90));
            va = v10.add(v10.sub(v20).dot(alfa));
            vb = v10.add(v20.sub(v10).Rot(90).dot(alfa));
            vc = v20.add(v20.sub(v10).dot(alfa));
            vd = v3.add(v20.sub(v10).Rot(-90).dot(alfa));

            a = new Point(Point.nextId(), va.x, va.y);
            b = new Point(Point.nextId(), vb.x, vb.y);
            c = new Point(Point.nextId(), vc.x, vc.y);
            d = new Point(Point.nextId(), vd.x, vd.y);

            p1 = new Point(Point.nextId(), v10.x, v10.y);
            p2 = new Point(Point.nextId(), v20.x, v20.y);
            p3 = new Point(Point.nextId(), v3.x, v3.y);
            setAssociateConstraints(null);
        }
        calculateDistance();
    }

    public String toString() {
        String out = type + "*" + this.primitiveId + ": {";
        out += "a =" + a + ",";
        out += "p1=" + p1 + ",";
        out += "b =" + b + ",";
        out += "c =" + c + ",";
        out += "p2=" + p2 + ",";
        out += "p3=" + p3 + ",";
        out += "d =" + d + "}\n";
        return out;
    }

    /**
     * Funkcja oblicza dlugosci poczatkowe pomiedzy wektorami
     */
    private void calculateDistance() {
        //Naciag wstepny lepiej sie zbiegaja
        d_a_p1 = Math.abs(p1.sub(a).length());
        d_b_p1 = Math.abs(p1.sub(b).length());
        d_p1_p2 = Math.abs(p2.sub(p1).length());
        d_p1_p3 = Math.abs(p3.sub(p1).length());
        d_p3_d = Math.abs(d.sub(p3).length());
        d_p2_c = Math.abs(c.sub(p2).length());
    }

    @Override
    public void evaluateGuidePoints() {
        Vector va = (Vector) p1.sub(p2.sub(p1).dot(alfa));
        Vector vb = (Vector) p1.sub(p3.sub(p1).unit().dot(p2.sub(p1).length()).dot(alfa));
        Vector vc = (Vector) p2.add(p2.sub(p1).dot(alfa));
        Vector vd = (Vector) p3.add(p3.sub(p1).unit().dot(p2.sub(p1).length()).dot(alfa));
        Vector v3 = (Vector) p1.add(p3.sub(p1).unit().dot(p2.sub(p1).length()));

        a.setLocation(va.x, va.y);
        b.setLocation(vb.x, vb.y);
        c.setLocation(vc.x, vc.y);
        d.setLocation(vd.x, vd.y);
        p3.setLocation(v3.x, v3.y);

        calculateDistance();

        //uaktulaniamy wiezy
        ((ConstraintFixPoint) Constraint.dbConstraint.get(constraints[0])).setFixVector(va);
        ((ConstraintFixPoint) Constraint.dbConstraint.get(constraints[1])).setFixVector(vb);
        ((ConstraintFixPoint) Constraint.dbConstraint.get(constraints[2])).setFixVector(vc);
        ((ConstraintFixPoint) Constraint.dbConstraint.get(constraints[3])).setFixVector(vd);
    }

    @Override
    public void setForce(int row, MatrixDouble mt) {
        Vector fap1 = p1.sub(a).unit().dot(Consts.springStiffnessLow).dot(p1.sub(a).length() - d_a_p1);
        Vector fbp1 = p1.sub(b).unit().dot(Consts.springStiffnessLow).dot(p1.sub(b).length() - d_b_p1);
        Vector fp1p2 = p2.sub(p1).unit().dot(Consts.springStiffnessHigh).dot(p2.sub(p1).length() - d_p1_p2);
        Vector fp1p3 = p3.sub(p1).unit().dot(Consts.springStiffnessHigh).dot(p3.sub(p1).length() - d_p1_p3);
        Vector fp2c = c.sub(p2).unit().dot(Consts.springStiffnessLow).dot(c.sub(p2).length() - d_p2_c);
        Vector fp3d = d.sub(p3).unit().dot(Consts.springStiffnessLow).dot(d.sub(p3).length() - d_p3_d);

        mt.setVector(row + 0, 0, fap1);
        mt.setVector(row + 2, 0, fbp1);
        mt.setVector(row + 4, 0, fp2c.dot(-1));
        mt.setVector(row + 6, 0, fp3d.dot(-1));
        mt.setVector(row + 8, 0, fp1p2.add(fp1p3).sub(fap1).sub(fbp1));
        mt.setVector(row + 10, 0, fp2c.sub(fp1p2));
        mt.setVector(row + 12, 0, fp3d.sub(fp1p3));
    }


    @Override
    public void setJacobian(int row, int col, MatrixDouble mt) {

        // K -mala sztywnosci
        MatrixDouble Kb = MatrixDouble.diagonal(Consts.springStiffnessHigh, Consts.springStiffnessHigh);
        MatrixDouble Ks = MatrixDouble.diagonal(Consts.springStiffnessLow, Consts.springStiffnessLow);

        MatrixDouble mKs = Ks.dotC(-1);
        MatrixDouble mKb = Kb.dotC(-1);
        MatrixDouble KsKbm = mKs.add(mKb);

        mt.addSubMatrix(row + 0, col + 0, mKs);
        mt.addSubMatrix(row + 0, col + 8, Ks);//a

        mt.addSubMatrix(row + 2, col + 2, mKs);
        mt.addSubMatrix(row + 2, col + 8, Ks);//b

        mt.addSubMatrix(row + 4, col + 4, mKs);
        mt.addSubMatrix(row + 4, col + 10, Ks);//c

        mt.addSubMatrix(row + 6, col + 6, mKs);
        mt.addSubMatrix(row + 6, col + 12, Ks);//d

        mt.addSubMatrix(row + 8, col + 0, Ks);
        mt.addSubMatrix(row + 8, col + 2, Ks);
        mt.addSubMatrix(row + 8, col + 8, KsKbm.dotC(2.0));
        mt.addSubMatrix(row + 8, col + 10, Kb);
        mt.addSubMatrix(row + 8, col + 12, Kb); //p1

        mt.addSubMatrix(row + 10, col + 4, Ks);
        mt.addSubMatrix(row + 10, col + 8, Kb);
        mt.addSubMatrix(row + 10, col + 10, KsKbm); //p2

        mt.addSubMatrix(row + 12, col + 6, Ks);
        mt.addSubMatrix(row + 12, col + 8, Kb);
        mt.addSubMatrix(row + 12, col + 12, KsKbm); //p3
    }

    @Override
    public void setAssociateConstraints(Set<Integer> skipIds) {
        if (skipIds == null) skipIds = Collections.emptySet();
        ConstraintFixPoint fixPointa = new ConstraintFixPoint(Constraint.nextId(skipIds), a, false);
        ConstraintFixPoint fixPointb = new ConstraintFixPoint(Constraint.nextId(skipIds), b, false);
        ConstraintFixPoint fixPointc = new ConstraintFixPoint(Constraint.nextId(skipIds), c, false);
        ConstraintFixPoint fixPointd = new ConstraintFixPoint(Constraint.nextId(skipIds), d, false);
        ConstraintEqualLength sameLength = new ConstraintEqualLength(Constraint.nextId(skipIds), p1, p2, p1, p3, false);

        //ConstraintLinesSameLength sameLength2= new ConstraintLinesSameLength(p2,c,p3,d);
        //ConstraintLinesParallelism par1 = new ConstraintLinesParallelism(a,p1,p2,c);
        //ConstraintLinesParallelism par2 = new ConstraintLinesParallelism(b,p1,p3,d);
        constraints = new int[5];
        constraints[0] = fixPointa.constraintId;
        constraints[1] = fixPointb.constraintId;
        constraints[2] = fixPointc.constraintId;
        constraints[3] = fixPointd.constraintId;
        constraints[4] = sameLength.constraintId;
        //constraintsId[5] = sameLength2.constraintId;
        //constraintsId[6] = par1.constraintId;
        //constraintsId[7] = par2.constraintId;
    }

    @Override
    public int getNumOfPoints() {
        return 7; //a,b,c,d,p1,p2,p3
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
        return p3.id;
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
        return c.id;
    }

    @Override
    public int getD() {
        return d.id;
    }


    @Override
    public int[] getAllPointsId() {
        int[] out = new int[7];
        out[0] = a.getId();
        out[1] = b.getId();
        out[2] = c.getId();
        out[3] = d.getId();
        out[4] = p1.getId();
        out[5] = p2.getId();
        out[6] = p3.getId();
        return out;
    }
}
