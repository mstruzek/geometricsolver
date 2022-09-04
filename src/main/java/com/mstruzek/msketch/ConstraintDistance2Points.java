package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.TensorDouble;

import static com.mstruzek.msketch.ModelRegistry.dbPoint;

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
        super(constId, ConstraintType.Distance2Points, true);
        k_id = K.id;
        l_id = L.id;
        param_id = param.getId();
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-Distance2Points" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " , Parametr-" + ModelRegistry.dbParameter.get(param_id).getId() + " = " + ModelRegistry.dbParameter.get(param_id).getValue() + " } \n";

    }

    @Override
    public void getJacobian(TensorDouble mts) {
        final Vector LKu = dbPoint.get(l_id).Vector().minus(dbPoint.get(k_id)).unit();
        TensorDouble mt = mts;
        int j = 0;
        //k
        j = po.get(k_id);
        mt.setVector(0, j * 2, LKu.product(-1.0));
        //l
        j = po.get(l_id);
        mt.setVector(0, j * 2, LKu);
    }

    @Override
    public boolean isJacobianConstant() {
        return true;
    }

    @Override
    public TensorDouble getValue() {
        final double value = dbPoint.get(l_id).minus(dbPoint.get(k_id)).length() - ModelRegistry.dbParameter.get(param_id).getValue();
        return TensorDouble.scalar(value);
    }

    @Override
    public void getHessian(TensorDouble mt, double lagrange) {
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
    public int getParameter() {
        return param_id;
    }

    @Override
    public double getNorm() {
        final TensorDouble md = getValue();
        return md.getQuick(0, 0);
    }
}
