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
public class ArcOld extends GeometricPrimitive {

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


    /** KOLEJNOSC INICJALIZACJI MA ZNACZENIE - PATRZ KONSTRUKTOR */

    /**
     * Odleglosci pomiedzy wektorami poczatkowe
     */
    double d_a_p1, d_p1_b, d_c_p2, d_p2_p3, d_p3_d;
    /**
     * wsp�czynnik do skalowania wzglednego dla wektorow
     */
    double alfa = 1.0;

    /**
     * Numery wiezow  powiazane z a,b
     */
    int[] constraintsId = new int[5];

    /**
     * Mnoznik sily
     */
    int dS = 10;

    /**
     * Luk
     *
     * @param p10 srodek okregu
     * @param p20 pierwszy koniec luku
     * @param p30 drugi koniec luku
     */
    public ArcOld(Vector p10, Vector p20, Vector p30) {
        super(ModelRegistry.nextPrimitiveId(), GeometricPrimitiveType.Arc);
        // FreePoint(a,p1,b)
        a = new Point(ModelRegistry.nextPointId(), p10.product(2).minus(p20));
        p1 = new Point(ModelRegistry.nextPointId(), p10);
        b = new Point(ModelRegistry.nextPointId(), p20);

        //  Line(c,p2,p3,d)
        Vector a = p20.minus(p10);
        c = new Point(ModelRegistry.nextPointId(), p10.plus(a.Rot(-90).product(3)));
        p2 = new Point(ModelRegistry.nextPointId(), p10.plus(a.Rot(-90)));
        p3 = new Point(ModelRegistry.nextPointId(), p10.plus(a.Rot(90)));
        d = new Point(ModelRegistry.nextPointId(), p10.plus(a.Rot(90).product(3)));
        calculateDistance();
        setAssociateConstraints(null);
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
        // FreePoint(a,p1,b)
        d_a_p1 = Math.abs(p1.minus(a).length());
        d_p1_b = Math.abs(b.minus(p1).length());

        // Line(c,p2,p3,d)
        d_c_p2 = Math.abs(p2.minus(c).length());
        d_p2_p3 = Math.abs(p3.minus(p2).length());
        d_p3_d = Math.abs(p3.minus(d).length());
    }

    @Override
    public void evaluateGuidePoints() {

        Vector pr = (Vector) (p3.minus(p2).Rot(-90).product(0.5));
        Vector va = (Vector) p1.plus(pr);
        Vector vb = (Vector) p1.minus(pr);
        //Line
        Vector vc = (Vector) p2.minus(p3.minus(p2));
        Vector vd = (Vector) p3.plus(p3.minus(p2));


        a.setLocation(va.getX(), va.getY());
        b.setLocation(vb.getX(), vb.getY());
        c.setLocation(vc.getX(), vc.getY());
        d.setLocation(vd.getX(), vd.getY());

        calculateDistance();

        //uaktulaniamy wiezy
        ((ConstraintFixPoint) constraints[0]).setFixVector(va);
        ((ConstraintFixPoint) constraints[1]).setFixVector(vb);
        ((ConstraintFixPoint) constraints[2]).setFixVector(vc);
        ((ConstraintFixPoint) constraints[3]).setFixVector(vd);

    }

    @Override
    public void evaluateForceIntensity(int row, MatrixDouble mt) {
        Vector f12 = p1.minus(a).unit().product(Consts.springStiffnessHigh * dS).product(p1.minus(a).length() - d_a_p1);        //F12 - sily w sprezynach
        Vector f23 = b.minus(p1).unit().product(Consts.springStiffnessHigh * dS).product(b.minus(p1).length() - d_p1_b);        //F23

        //FREEPOINT
        mt.setVector(row + 0, 0, f12);             //F1 - silu na poszczegolne punkty
        mt.setVector(row + 2, 0, f23.minus(f12));    //F2
        mt.setVector(row + 4, 0, f23.product(-1));     //F3

        Vector fcp2 = p2.minus(c).unit().product(Consts.springStiffnessLow).product(p2.minus(c).length() - d_c_p2);         //LINE
        Vector fp2p3 = p3.minus(p2).unit().product(Consts.springStiffnessHigh).product(p3.minus(p2).length() - d_p2_p3);    //F23
        Vector fp3d = d.minus(p3).unit().product(Consts.springStiffnessLow).product(d.minus(p3).length() - d_p3_d);

        mt.setVector(row + 6, 0, fcp2);
        mt.setVector(row + 7, 0, fp2p3.minus(fcp2));
        mt.setVector(row + 10, 0, fp3d.minus(fp2p3));
        mt.setVector(row + 12, 0, fp3d.product(-1));
    }


    @Override
    public void setStiffnessMatrix(int row, int col, MatrixDouble mt) {

        /** Free Point
         * k= I*k
         * [ -ks    ks     0;
         *    ks  -2ks   ks ;
         *     0    ks   -ks];

         */
        /** Line -shift = 6
         * k= I*k
         * [ -ks    ks      0  	  0;
         *    ks  -ks-kb   kb  	  0;
         *     0    kb   -ks-kb   ks;
         *     0  	 0     ks    -ks];
         */

        // K -mala sztywnosci
        MatrixDouble Ksp = MatrixDouble.diagonal(Consts.springStiffnessHigh * dS, Consts.springStiffnessHigh * dS);
        MatrixDouble Ksm = Ksp.multiplyC(-1);
        MatrixDouble Ks = MatrixDouble.diagonal(Consts.springStiffnessLow, Consts.springStiffnessLow);
        MatrixDouble Kb = MatrixDouble.diagonal(Consts.springStiffnessHigh, Consts.springStiffnessHigh);
        // -Ks-Kb
        MatrixDouble Ksb = Ks.multiplyC(-1).plusSubMatrix(0, 0, Kb.multiplyC(-1));

        //FREEPOINT
        mt.plusSubMatrix(row + 0, col + 0, Ksm);
        mt.plusSubMatrix(row + 0, col + 2, Ksp);
        mt.plusSubMatrix(row + 2, col + 0, Ksp);
        mt.plusSubMatrix(row + 2, col + 2, Ksp.multiplyC(2.0));
        mt.plusSubMatrix(row + 2, col + 4, Ksp);
        mt.plusSubMatrix(row + 4, col + 2, Ksp);
        mt.plusSubMatrix(row + 4, col + 4, Ksm);

        //LINE
        int s = 6;//przesuniecie
        mt.plusSubMatrix(row + 0 + s, col + 0 + s, Ks.multiplyC(-1));
        mt.plusSubMatrix(row + 0 + s, col + 2 + s, Ks);
        mt.plusSubMatrix(row + 2 + s, col + 0 + s, Ks);
        mt.plusSubMatrix(row + 2 + s, col + 2 + s, Ksb.multiplyC(2.0));
        mt.plusSubMatrix(row + 2 + s, col + 4 + s, Kb);
        mt.plusSubMatrix(row + 4 + s, col + 2 + s, Kb);
        mt.plusSubMatrix(row + 4 + s, col + 4 + s, Ksb);
        mt.plusSubMatrix(row + 4 + s, col + 6 + s, Ks);
        mt.plusSubMatrix(row + 6 + s, col + 4 + s, Ks);
        mt.plusSubMatrix(row + 6 + s, col + 6 + s, Ks.multiplyC(-1));
    }

    @Override
    public void setAssociateConstraints(Set<Integer> skipIds) {
        if (skipIds == null) skipIds = Collections.emptySet();
        ConstraintFixPoint fixPointa = new ConstraintFixPoint(ModelRegistry.nextConstraintId(skipIds), a, false);
        ConstraintFixPoint fixPointb = new ConstraintFixPoint(ModelRegistry.nextConstraintId(skipIds), b, false);
        ConstraintFixPoint fixPointc = new ConstraintFixPoint(ModelRegistry.nextConstraintId(skipIds), c, false);
        ConstraintFixPoint fixPointd = new ConstraintFixPoint(ModelRegistry.nextConstraintId(skipIds), d, false);
        ConstraintEqualLength sameLength = new ConstraintEqualLength(ModelRegistry.nextConstraintId(skipIds), p1, p2, p1, p3, false);
        constraints = new Constraint[5];
        constraints[0] = fixPointa;
        constraints[1] = fixPointb;
        constraints[2] = fixPointc;
        constraints[3] = fixPointd;
        constraints[4] = sameLength;
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
        out[1] = p1.getId();
        out[2] = b.getId();
        out[3] = c.getId();
        out[4] = p2.getId();
        out[5] = p3.getId();
        out[6] = d.getId();
        return out;
    }

    @Override
    public Point[] getAllPoints() {
        return new Point[]{ a, p1, b, c, p2, p3, d };
    }
}