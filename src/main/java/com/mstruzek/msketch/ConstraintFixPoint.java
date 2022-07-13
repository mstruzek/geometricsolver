package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

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

    /**
     * Konstruktor wiezu
     *
     * @param K - zrzutowany Point na Vector
     * @param L - Vector w ktorym bedzie zafiksowany K
     */
    public ConstraintFixPoint(Point K, Vector L) {
        super(nextId(), GeometricConstraintType.FixPoint, true);
        //pobierz id
        k_id = ((Point) K).id;
        k0_vec = L;
        dbConstraint.put(constraintId, this);
    }

    public ConstraintFixPoint(Integer constId,Point K,boolean storage){
        super(constId, GeometricConstraintType.FixPoint, storage);
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
        MatrixDouble mt = getValue();
        double norm = Matrix.constructWithCopy(mt.getArray()).norm1();

        return "Constraint-FixPoint" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  , K0 = " + k0_vec + " } \n";
    }

    @Override
    public MatrixDouble getJacobian() {
        int j = 0;
        MatrixDouble mt = MatrixDouble.fill(2, dbPoint.size() * 2, 0.0);
        for(Integer i : dbPoint.keySet()) {
            if(k_id == dbPoint.get(i).id) {
                //macierz jednostkowa
                mt.setSubMatrix(0, j * 2 , MatrixDouble.identity(2));
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
        return new MatrixDouble(dbPoint.get(k_id).sub(k0_vec), true);
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
    public int getParametr() {
        return -1;
    }

    @Override
    public double getNorm() {
        MatrixDouble mt = getValue();
        double val = Math.sqrt(mt.get(0,0) * mt.get(0,0) + mt.get(1,0) * mt.get(1,0));
        return val;
    }
}
