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
        this(ModelRegistry.nextPrimitiveId(), v10, v20, v30);
    }

    public Arc(int id, Vector v10, Vector v20, Vector v30) {
        super(id, GeometricPrimitiveType.Arc);

        // FreePoint(a,p1,b)
        Vector va, vb, vc, vd, v3;


        if (v10 instanceof Point && v20 instanceof Point) {

            p1 = (Point) v10;
            p2 = (Point) v20;
            p3 = (Point) v30;

            va = v10.plus(v10.minus(v20).product(alfa));
            vb = v10.plus(v20.minus(v10).Rot(90).product(alfa));
            vc = v20.plus(v20.minus(v10).product(alfa));
            vd = v30.plus(v20.minus(v10).Rot(-90).product(alfa));

            a = new Point(p1.getId() - 4, va);
            b = new Point(p1.getId() - 3, vb);
            c = new Point(p1.getId() - 2, vc);
            d = new Point(p1.getId() - 1, vd);

        } else {
            v3 = v10.plus(v20.minus(v10).Rot(-90));
            va = v10.plus(v10.minus(v20).product(alfa));
            vb = v10.plus(v20.minus(v10).Rot(90).product(alfa));
            vc = v20.plus(v20.minus(v10).product(alfa));
            vd = v3.plus(v20.minus(v10).Rot(-90).product(alfa));

            a = new Point(ModelRegistry.nextPointId(), va);
            b = new Point(ModelRegistry.nextPointId(), vb);
            c = new Point(ModelRegistry.nextPointId(), vc);
            d = new Point(ModelRegistry.nextPointId(), vd);

            p1 = new Point(ModelRegistry.nextPointId(), v10);
            p2 = new Point(ModelRegistry.nextPointId(), v20);
            p3 = new Point(ModelRegistry.nextPointId(), v3);
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
        d_a_p1 = Math.abs(p1.minus(a).length());
        d_b_p1 = Math.abs(p1.minus(b).length());
        d_p1_p2 = Math.abs(p2.minus(p1).length());
        d_p1_p3 = Math.abs(p3.minus(p1).length());
        d_p3_d = Math.abs(d.minus(p3).length());
        d_p2_c = Math.abs(c.minus(p2).length());
    }

    @Override
    public void evaluateGuidePoints() {
        Vector va = (Vector) p1.minus(p2.minus(p1).product(alfa));
        Vector vb = (Vector) p1.minus(p3.minus(p1).unit().product(p2.minus(p1).length()).product(alfa));
        Vector vc = (Vector) p2.plus(p2.minus(p1).product(alfa));
        Vector vd = (Vector) p3.plus(p3.minus(p1).unit().product(p2.minus(p1).length()).product(alfa));
        Vector v3 = (Vector) p1.plus(p3.minus(p1).unit().product(p2.minus(p1).length()));

        a.setLocation(va.getX(), va.getY());
        b.setLocation(vb.getX(), vb.getY());
        c.setLocation(vc.getX(), vc.getY());
        d.setLocation(vd.getX(), vd.getY());
        p3.setLocation(v3.getX(), v3.getY());

        calculateDistance();

        //uaktulaniamy wiezy
        ((ConstraintFixPoint) constraints[0]).setFixVector(va);
        ((ConstraintFixPoint) constraints[1]).setFixVector(vb);
        ((ConstraintFixPoint) constraints[2]).setFixVector(vc);
        ((ConstraintFixPoint) constraints[3]).setFixVector(vd);
    }

    @Override
    public void evaluateForceIntensity(int row, MatrixDouble mt) {
        Vector fap1 = p1.minus(a).unit().product(Consts.springStiffnessLow).product(p1.minus(a).length() - d_a_p1);
        Vector fbp1 = p1.minus(b).unit().product(Consts.springStiffnessLow).product(p1.minus(b).length() - d_b_p1);
        Vector fp1p2 = p2.minus(p1).unit().product(Consts.springStiffnessHigh).product(p2.minus(p1).length() - d_p1_p2);
        Vector fp1p3 = p3.minus(p1).unit().product(Consts.springStiffnessHigh).product(p3.minus(p1).length() - d_p1_p3);
        Vector fp2c = c.minus(p2).unit().product(Consts.springStiffnessLow).product(c.minus(p2).length() - d_p2_c);
        Vector fp3d = d.minus(p3).unit().product(Consts.springStiffnessLow).product(d.minus(p3).length() - d_p3_d);

        mt.setVector(row + 0, 0, fap1);
        mt.setVector(row + 2, 0, fbp1);
        mt.setVector(row + 4, 0, fp2c.product(-1));
        mt.setVector(row + 6, 0, fp3d.product(-1));
        mt.setVector(row + 8, 0, fp1p2.plus(fp1p3).minus(fap1).minus(fbp1));
        mt.setVector(row + 10, 0, fp2c.minus(fp1p2));
        mt.setVector(row + 12, 0, fp3d.minus(fp1p3));
    }


    @Override
    public void setStiffnessMatrix(int row, int col, MatrixDouble mt) {

        // K -mala sztywnosci
        MatrixDouble Kb = MatrixDouble.diagonal(Consts.springStiffnessHigh, Consts.springStiffnessHigh);
        MatrixDouble Ks = MatrixDouble.diagonal(Consts.springStiffnessLow, Consts.springStiffnessLow);

        MatrixDouble mKs = Ks.multiplyC(-1);
        MatrixDouble mKb = Kb.multiplyC(-1);
        MatrixDouble KsKbm = mKs.plus(mKb);

        mt.plusSubMatrix(row + 0, col + 0, mKs);
        mt.plusSubMatrix(row + 0, col + 8, Ks);//a

        mt.plusSubMatrix(row + 2, col + 2, mKs);
        mt.plusSubMatrix(row + 2, col + 8, Ks);//b

        mt.plusSubMatrix(row + 4, col + 4, mKs);
        mt.plusSubMatrix(row + 4, col + 10, Ks);//c

        mt.plusSubMatrix(row + 6, col + 6, mKs);
        mt.plusSubMatrix(row + 6, col + 12, Ks);//d

        mt.plusSubMatrix(row + 8, col + 0, Ks);
        mt.plusSubMatrix(row + 8, col + 2, Ks);
        mt.plusSubMatrix(row + 8, col + 8, KsKbm.multiplyC(2.0));
        mt.plusSubMatrix(row + 8, col + 10, Kb);
        mt.plusSubMatrix(row + 8, col + 12, Kb); //p1

        mt.plusSubMatrix(row + 10, col + 4, Ks);
        mt.plusSubMatrix(row + 10, col + 8, Kb);
        mt.plusSubMatrix(row + 10, col + 10, KsKbm); //p2

        mt.plusSubMatrix(row + 12, col + 6, Ks);
        mt.plusSubMatrix(row + 12, col + 8, Kb);
        mt.plusSubMatrix(row + 12, col + 12, KsKbm); //p3
    }

    @Override
    public void setAssociateConstraints(Set<Integer> skipIds) {
        if (skipIds == null) skipIds = Collections.emptySet();
        ConstraintFixPoint fixPointa = new ConstraintFixPoint(ModelRegistry.nextConstraintId(skipIds), a, false);
        ConstraintFixPoint fixPointb = new ConstraintFixPoint(ModelRegistry.nextConstraintId(skipIds), b, false);
        ConstraintFixPoint fixPointc = new ConstraintFixPoint(ModelRegistry.nextConstraintId(skipIds), c, false);
        ConstraintFixPoint fixPointd = new ConstraintFixPoint(ModelRegistry.nextConstraintId(skipIds), d, false);
        ConstraintEqualLength sameLength = new ConstraintEqualLength(ModelRegistry.nextConstraintId(skipIds), p1, p2, p1, p3, false);

        //ConstraintLinesSameLength sameLength2= new ConstraintLinesSameLength(p2,c,p3,d);
        //ConstraintLinesParallelism par1 = new ConstraintLinesParallelism(a,p1,p2,c);
        //ConstraintLinesParallelism par2 = new ConstraintLinesParallelism(b,p1,p3,d);
        constraints = new Constraint[5];
        constraints[0] = fixPointa;
        constraints[1] = fixPointb;
        constraints[2] = fixPointc;
        constraints[3] = fixPointd;
        constraints[4] = sameLength;
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

    @Override
    public Point[] getAllPoints() {
        return new Point[]{ a, b, c, d, p1, p2, p3 };
    }
}
