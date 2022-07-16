package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

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
        super(constId, GeometricConstraintType.ParametrizedXFix, persistent);
        this.k_id = K.id;
        this.param_id = parameter.getId();
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-ParametrizedXFix" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  , P = " + Parameter.dbParameter.get(param_id) + " } \n";
    }

    @Override
    public MatrixDouble getJacobian() {
        int j = 0;
        MatrixDouble mt = MatrixDouble.matrix1D(dbPoint.size() * 2, 0.0);
        for (Integer i : dbPoint.keySet()) {
            if (k_id == dbPoint.get(i).id) {
                /// wspolrzedna [X]
                mt.set(0, j * 2, 1.0);
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
        double value = dbPoint.get(k_id).getX() - Parameter.dbParameter.get(param_id).getValue();
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
