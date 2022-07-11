package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

public class ConstraintDistancePointLine extends Constraint {

    /** Punkty kontrolne */
    /** Point K-id */
    int k_id;
    /** Point L-id */
    int l_id;
    /** Point M-id */
    int m_id;
    /** Numer parametru przechowujacy kat w radianach */
    int param_id;

    /**
     * Konstruktor pomiedzy 3 punktami i paramtetrem
     * rownanie tego wiezu to [(R(L-K))'*(M-K)]^2 - d*d*(L-K)'*(L-K) = 0  gdzie R = Rot(PI/2) = [ 0 -1 ; 1 0]
     *
     * @param constId
     * @param K       punkt prowadzacy prostej
     * @param L       punkt prowadzacy prostej
     * @param M       punkt odlegly od prowadzacej
     */
    public ConstraintDistancePointLine(int constId,Point K,Point L ,Point M,Parameter param){
        super(constId, GeometricConstraintType.DistancePointLine, true);
        k_id = K.id;
        l_id = L.id;
        m_id = M.id;
        param_id =param.getId();
    }

    public String toString(){
        MatrixDouble mt = getValue();
        double norm = Matrix.constructWithCopy(mt.getArray()).norm1();
        return "Constraint-DistancePointLine" + constraintId + "*s" + size() + " = " + norm  + " { K =" + Point.dbPoint.get(k_id) + "  ,L =" + Point.dbPoint.get(l_id) + " ,M =" + Point.dbPoint.get(m_id) + ", Parametr-" + Parameter.dbParameter.get(param_id).getId() + " = "+Parameter.dbParameter.get(param_id).getValue() + "} \n";
    }

    @Override
    public MatrixDouble getValue() {
        return new MatrixDouble(1,1);
    }

    @Override
    public MatrixDouble getJacobian() {
        return null;
    }

    @Override
    public double getNorm() {
        return 0;
    }

    @Override
    public boolean isJacobianConstant() {
        return false;
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
        return m_id;
    }

    @Override
    public int getN() {
        return -1;
    }

    @Override
    public int getParametr() {
        return param_id;
    }
}
