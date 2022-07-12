package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Klasa reprezentujaca wiez typu polaczenie
 * dwoch punktow(wektorow uogulnionych) PointK[y]-PoinL[y]=0;
 *
 * @author root
 */
public class ConstraintConnectVertical extends Constraint {

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
    public ConstraintConnectVertical(int constId, Point K, Point L) {
        super(constId, GeometricConstraintType.ConnectVertical, true);
        //pobierz id
        k_id = K.id;
        l_id = L.id;
    }

    public String toString() {
        MatrixDouble out = getValue();
        double norm = Matrix.constructWithCopy(out.getArray()).norm1();
        return "Constraint-ConnectVertical" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  , L = " + dbPoint.get(l_id) + " } \n";
    }

    @Override
    public MatrixDouble getJacobian() {
        MatrixDouble mt = MatrixDouble.fill(1, dbPoint.size() * 2, 0.0);
        int j = 0;
        for (Integer i : dbPoint.keySet()) {
            if (k_id == dbPoint.get(i).id) {
                mt.set(0, j * 2 + 1, 1.0);         // zero-Y
            }
            if (l_id == dbPoint.get(i).id) {
                mt.set(0, j * 2 + 1, -1.0);       // zero-Y
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
        MatrixDouble mt = new MatrixDouble(1, 1);
        mt.set(0, 0, dbPoint.get(k_id).getY() - dbPoint.get(l_id).getY());
        return mt;
    }

    @Override
    public MatrixDouble getHessian(double alfa) {
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
    public int getParametr() {
        return -1;
    }

    @Override
    public double getNorm() {
        MatrixDouble md = getValue();
        return md.get(0, 0);
    }
}
