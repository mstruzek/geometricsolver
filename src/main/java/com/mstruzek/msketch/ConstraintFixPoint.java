package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;
import com.mstruzek.msketch.matrix.MatrixDouble2D;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Klasa reprezentuje wiez typu FIX : "PointK - VectorK0 =0";
 *
 * @author root
 */
public class ConstraintFixPoint extends Constraint {

    /**
     * Point K-id
     */
    int k_id;
    /**
     * Vector L
     */
    Vector k0_vec = null;

    /**
     * Konstruktor wiezu
     *
     * @param constId
     * @param K       - Point zafiksowany bedzie w aktualnym miejscu
     */
    public ConstraintFixPoint(int constId, Point K) {
        this(constId, K, true);
    }

    public ConstraintFixPoint(Integer constId, Point K, boolean persistent) {
        super(constId, GeometricConstraintType.FixPoint, persistent);
        this.k_id = K.id;
        this.k0_vec = new Vector(K.x, K.y);
    }

    /**
     * Funkcja ustawia nowy wektor zafiksowania
     *
     * @param vc
     */
    public void setFixVector(Vector vc) {
        k0_vec = vc;
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-FixPoint" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  , K0 = " + k0_vec + " } \n";
    }

    @Override
    public MatrixDouble getJacobian() {
        int j = 0;
        MatrixDouble mt = MatrixDouble.matrix2D(2, dbPoint.size() * 2, 0.0);
        for (Integer i : dbPoint.keySet()) {
            if (k_id == dbPoint.get(i).id) {
                //macierz jednostkowa
                mt.setSubMatrix(0, j * 2, MatrixDouble.identity(2, 1.0));
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
        return new MatrixDouble2D(dbPoint.get(k_id).sub(k0_vec), true);
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
        return -1;
    }

    @Override
    public double getNorm() {
        MatrixDouble mt = getValue();
        double val = Math.sqrt(mt.get(0, 0) * mt.get(0, 0) + mt.get(1, 0) * mt.get(1, 0));
        return val;
    }
}
