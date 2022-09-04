package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.TensorDouble;

import static com.mstruzek.msketch.ModelRegistry.dbPoint;

/**
 * Klasa reprezentuje wiez typu Parametrized X coordinate : "PointK[x] - Parameter[x] = 0";
 *
 * @author root
 */
public class ConstraintParametrizedXFix extends Constraint {

    /*** Point K-id */
    int k_id;

    /*** Numer parametru przechowujacy wartosc polozenia [X] */
    int param_id;

    /**
     * Konstruktor wiezu
     *
     * @param constId
     * @param K
     */
    public ConstraintParametrizedXFix(int constId, Point K, Parameter parameter) {
        this(constId, K, parameter, true);
    }

    public ConstraintParametrizedXFix(Integer constId, Point K, Parameter parameter, boolean persistent) {
        super(constId, ConstraintType.ParametrizedXFix, persistent);
        this.k_id = K.id;
        this.param_id = parameter.getId();
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-ParametrizedXFix" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  , P = " + ModelRegistry.dbParameter.get(param_id) + " } \n";
    }

    @Override
    public void getJacobian(TensorDouble mts) {
        TensorDouble mt = mts;
        int j;
        //k
        j = po.get(k_id);
        /// wspolrzedna [X]
        mt.setQuick(0, j * 2, 1.0);
    }

    @Override
    public boolean isJacobianConstant() {
        return true;
    }

    @Override
    public TensorDouble getValue() {
        double value = dbPoint.get(k_id).getX() - ModelRegistry.dbParameter.get(param_id).getValue();
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
