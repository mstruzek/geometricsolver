package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Klasa reprezentujaca wiez typu polaczenie
 * dwoch punktow(wektorow uogulnionych) PointK[x]-PoinL[x]=0;
 *
 * @author root
 */
public class ConstraintHorizontalPoint extends Constraint {

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
    public ConstraintHorizontalPoint(int constId, Point K, Point L) {
        super(constId, GeometricConstraintType.HorizontalPoint, true);
        //pobierz id
        k_id = K.id;
        l_id = L.id;
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-Conect2Points" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  , L = " + dbPoint.get(l_id) + " } \n";
    }

    @Override
    public MatrixDouble getJacobian() {
        MatrixDouble mt = MatrixDouble.matrix1D(dbPoint.size() * 2, 0.0);
        int j = 0;
        for (Integer i : dbPoint.keySet()) {
            if (k_id == dbPoint.get(i).id) {
                mt.set(0, j * 2, 1.0);         // zero-X
            }
            if (l_id == dbPoint.get(i).id) {
                mt.set(0, j * 2, -1.0);       // zero-X
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
        double value = dbPoint.get(k_id).getX() - dbPoint.get(l_id).getX();
        return MatrixDouble.scalar(value);
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
        return md.get(0, 0);
    }
}
