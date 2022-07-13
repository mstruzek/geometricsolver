package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Parameter.dbParameter;
import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Klasa reprezentujaca wiez odleglosci pomiedzy 2 punktami
 *
 * @author root
 */
public class ConstraintDistance2Points extends Constraint {

    /** Punkty kontrolne */
    /**
     * Point K-id
     */
    int k_id;
    /**
     * Point L-id
     */
    int l_id;
    /**
     * Numer parametru
     */
    int param_id;

    /**
     * Konstruktor pomiedzy 2 punktami
     * rownanie tego wiezu to sqrt[(K-L)'*(K-L)] - d = 0
     *
     * @param constId
     * @param K
     * @param L
     */
    public ConstraintDistance2Points(int constId, Point K, Point L, Parameter param) {
        super(constId, GeometricConstraintType.Distance2Points, true);
        k_id = K.id;
        l_id = L.id;
        param_id = param.getId();
    }

    public String toString() {
        MatrixDouble mt = getValue();
        double norm = Matrix.constructWithCopy(mt.getArray()).norm1();
        return "Constraint-Distance2Points" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " , Parametr-" + dbParameter.get(param_id).getId() + " = " + dbParameter.get(param_id).getValue() + " } \n";

    }

    @Override
    public MatrixDouble getJacobian() {
        /// macierz 1xN
        MatrixDouble mt = MatrixDouble.fill(1, dbPoint.size() * 2, 0.0);
        Vector LK = dbPoint.get(l_id).Vector().sub(dbPoint.get(k_id)).unit();
        int j = 0;
        for (Integer i : dbPoint.keySet()) {
            if (k_id == dbPoint.get(i).id) {
                mt.setVectorT(0, j * 2, LK.dot(-1.0));
            }
            if (l_id == dbPoint.get(i).id) {
                mt.setVectorT(0, j * 2, LK);
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
        mt.set(0, 0, dbPoint.get(l_id).sub(dbPoint.get(k_id)).length() - dbParameter.get(param_id).getRadians());
        return mt;
    }

    @Override
    public MatrixDouble getHessian(double lagrange) {
        return null;
    }

    @Override
    public boolean isHessianConst() {
        return true;
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
        return param_id;
    }

    @Override
    public double getNorm() {
        MatrixDouble md = getValue();
        return md.get(0, 0);
    }
}
