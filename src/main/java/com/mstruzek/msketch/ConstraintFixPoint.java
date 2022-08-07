package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.TensorDouble;

import static com.mstruzek.msketch.ModelRegistry.dbPoint;

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
        this.k0_vec = new Vector(K.getX(), K.getY());
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
    public void getJacobian(TensorDouble mts) {
        TensorDouble mt = mts;
        int j;
        //k
        j = po.get(k_id);
        mt.setSubMatrix(0, j * 2, TensorDouble.identity(2, 1.0));
    }

    @Override
    public boolean isJacobianConstant() {
        return true;
    }

    @Override
    public TensorDouble getValue() {
        return TensorDouble.smallMatrix(dbPoint.get(k_id).minus(k0_vec), true);
    }

    @Override
    public TensorDouble getHessian(double lagrange) {
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
        TensorDouble mt = getValue();
        double val = Math.sqrt(mt.getQuick(0, 0) * mt.getQuick(0, 0) + mt.getQuick(1, 0) * mt.getQuick(1, 0));
        return val;
    }
}
