package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Klasa reprezentujaca wiez typu polaczenie
 * dwoch punktow(wektorow uogulnionych) PointK-PoinL=0;
 *
 * @author root
 */
public class ConstraintConnect2Points extends Constraint {

    /**
     * Point K-id
     */
    int k_id;
    /**
     * Point L-id
     */
    int l_id;

    /**
     * Konstruktor wiezu
     *
     * @param constId
     * @param K       - zrzutowany Point na Vector
     * @param L       - Vector w ktorym bedzie zafiksowany K
     */
    public ConstraintConnect2Points(int constId, Point K, Point L) {
        super(constId, GeometricConstraintType.Connect2Points, true);
        //pobierz id
        k_id = K.id;
        l_id = L.id;
    }


    public String toString() {
        MatrixDouble out = getValue();
        double norm = Matrix.constructWithCopy(out.getArray()).norm1();
        return "Constraint-Conect2Points" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  , L = " + dbPoint.get(l_id) + " } \n";
    }

    @Override
    public MatrixDouble getJacobian() {
        MatrixDouble mt = MatrixDouble.fill(2, dbPoint.size() * 2, 0.0);
        int j = 0;
        for (Integer i : dbPoint.keySet()) {
            if (k_id == dbPoint.get(i).id) {
                mt.setSubMatrix(0, j * 2, MatrixDouble.diag(2, 1.0));        //macierz jednostkowa = I
            }
            if (l_id == dbPoint.get(i).id) {
                mt.setSubMatrix(0, j * 2, MatrixDouble.diag(2, -1.0));       // = -I
            }
            j++;
        }
        return mt;
    }

    @Override
    public boolean isJacobianConstant() {
        return true;
    }


    @Override
    public MatrixDouble getValue() {
        return new MatrixDouble(dbPoint.get(k_id).Vector().sub(dbPoint.get(l_id)), true);
    }

    @Override
    public MatrixDouble getHessian(double lagrange) {
        return null;
    }

    @Override
    public boolean isHessianConst() {
        return false;
    }

    @Override
    public int getK() {
        return k_id;
    }

    @Override
    public int getL() {
        return l_id;
    }

    @Override
    public int getM() {
        return -1;
    }

    @Override
    public int getN() {
        return -1;
    }

    @Override
    public int getParameter() {
        return -1;
    }

    @Override
    public double getNorm() {
        MatrixDouble md = getValue();
        return Math.sqrt(md.get(0, 0) * md.get(0, 0) + md.get(1, 0) * md.get(1, 0));
    }

}
