package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

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
        super(constId, GeometricConstraintType.ParametrizedYFix, persistent);
        this.k_id = K.id;
        this.param_id = parameter.getId();
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-ParametrizedYFix" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  , P = " + Parameter.dbParameter.get(param_id) + " } \n";
    }

    @Override
    public MatrixDouble getJacobian() {
        int j = 0;
        MatrixDouble mt = MatrixDouble.fill(1, Point.dbPoint.size() * 2, 0.0);
        for (Integer i : Point.dbPoint.keySet()) {
            if (k_id == Point.dbPoint.get(i).id) {
                /// wspolrzedna [Y]
                mt.set(0, j * 2 + 1, 1.0);
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
        mt.set(0, 0, dbPoint.get(k_id).getY() - Parameter.dbParameter.get(param_id).getValue());
        return mt;
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
        MatrixDouble mt = getValue();
        return mt.get(0, 0);
    }
}
