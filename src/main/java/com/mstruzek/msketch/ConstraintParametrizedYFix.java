package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.TensorDouble;

import static com.mstruzek.msketch.ModelRegistry.dbPoint;

/**
 * Klasa reprezentuje wiez typu Parametrized Y coordinate : "PointK[y] - Parameter[y] = 0";
 *
 * @author root
 */
public class ConstraintParametrizedYFix extends Constraint {

    /*** Point K-id */
    int k_id;

    /*** Numer parametru przechowujacy wartosc polozenia [Y] */
    int param_id;

    /**
     * Konstruktor wiezu
     *
     * @param constId
     * @param K
     */
    public ConstraintParametrizedYFix(int constId, Point K, Parameter parameter) {
        this(constId, K, parameter, true);
    }

    public ConstraintParametrizedYFix(Integer constId, Point K, Parameter parameter, boolean persistent) {
        super(constId, ConstraintType.ParametrizedYFix, persistent);
        this.k_id = K.id;
        this.param_id = parameter.getId();
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-ParametrizedYFix" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  , P = " + ModelRegistry.dbParameter.get(param_id) + " } \n";
    }

    @Override
    public void getJacobian(TensorDouble mts) {
        TensorDouble mt = mts;
        int j;
        // K
        j = po.get(k_id);
        // wspolrzedna [Y]
        mt.setQuick(0, j * 2 + 1, 1.0);
    }

    @Override
    public boolean isJacobianConstant() {
        return true;
    }

    @Override
    public TensorDouble getValue() {
        double value = dbPoint.get(k_id).getY() - ModelRegistry.dbParameter.get(param_id).getValue();
        return TensorDouble.scalar(value);
    }

    @Override
    public void getHessian(TensorDouble mt, double lagrange) {
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
        return -1;
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
        TensorDouble mt = getValue();
        return mt.getQuick(0, 0);
    }
}
