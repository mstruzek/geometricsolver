package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Parameter.dbParameter;
import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Wiez odpowiedzialny za kat pomiedzy wektorami
 *
 * @author root
 */
public class ConstraintAngle2Lines extends Constraint {

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
     * Point M-id
     */
    int m_id;
    /**
     * Point N-id
     */
    int n_id;
    /**
     * Numer parametru przechowujacy kat w radianach
     */
    int param_id;


    /**
     * Konstruktor pomiedzy 4 punktami i paramtetrem
     * rownanie tego wiezu to (L-K)'*(N-M)-cos(param)*|L-K|*|N-M| = 0
     *
     * @param K punkt prostej
     * @param L punkt prostej
     * @param M punkt prostej
     * @param N punkt prostej
     */
    public ConstraintAngle2Lines(Integer constId, Point K, Point L, Point M, Point N, Parameter param) {
        super(constId, GeometricConstraintType.Angle2Lines, true);
        k_id = K.id;
        l_id = L.id;
        m_id = M.id;
        n_id = N.id;
        param_id = param.getId();
    }

    public String toString() {
        MatrixDouble mt = getValue();
        double norm = Matrix.constructWithCopy(mt.getArray()).norm1();
        return "Constraint-Angle2Lines" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) + ", Parametr-" + dbParameter.get(param_id).getId() + " = " + dbParameter.get(param_id).getValue() + "} \n";
    }

    @Override
    public MatrixDouble getValue() {
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
        MatrixDouble mt = new MatrixDouble(1, 1);
        mt.set(0, 0, LK.dot(NM) - LK.length() * NM.length() * Math.cos(dbParameter.get(param_id).getRadians()));
        return mt;
    }

    @Override
    public MatrixDouble getJacobian() {
        MatrixDouble mt = MatrixDouble.fill(1, dbPoint.size() * 2, 0.0);
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
        Vector uLKdNM = LK.unit().dot(NM.length()).dot(Math.cos(dbParameter.get(param_id).getRadians()));
        Vector uNMdLK = NM.unit().dot(LK.length()).dot(Math.cos(dbParameter.get(param_id).getRadians()));
        int j = 0;
        for (Integer i : dbPoint.keySet()) {
            if (k_id == dbPoint.get(i).id) {
                mt.set(0, j * 2     , -NM.x + uLKdNM.x);
                mt.set(0, j * 2 + 1 , -NM.y + uLKdNM.y);
            }
            if (l_id == dbPoint.get(i).id) {
                mt.set(0, j * 2     , NM.x - uLKdNM.x);
                mt.set(0, j * 2 + 1 , NM.y - uLKdNM.y);
            }
            if (m_id == dbPoint.get(i).id) {
                mt.set(0, j * 2     , -LK.x + uNMdLK.x);
                mt.set(0, j * 2 + 1 , -LK.y + uNMdLK.y);
            }
            if (n_id == dbPoint.get(i).id) {
                mt.set(0, j * 2     , LK.x - uNMdLK.x);
                mt.set(0, j * 2 + 1 , LK.y - uNMdLK.y);
            }
            j++;
        }
        return mt;
    }

    @Override
    public boolean isJacobianConstant() {
        return false;
    }

    @Override
    public MatrixDouble getHessian(double lagrange) {
        MatrixDouble mt = MatrixDouble.fill(dbPoint.size() * 2, dbPoint.size() * 2, 0.0);
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id)).unit();
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id)).unit();
        double g = LK.dot(NM) * Math.cos(dbParameter.get(param_id).getRadians());
        int i = 0;
        for (Integer pI : dbPoint.keySet()) { /// wiersz
            int j = 0;
            for (Integer pJ : dbPoint.keySet()) { /// kolumna
                //k,k
                if (k_id == dbPoint.get(pI).id && k_id == dbPoint.get(pJ).id) {
                    // 0
                }
                //k,l
                if (k_id == dbPoint.get(pI).id && l_id == dbPoint.get(pJ).id) {
                    //0
                }
                //k,m
                if (k_id == dbPoint.get(pI).id && m_id == dbPoint.get(pJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, 1 - g).dot(lagrange));
                }
                //k,n
                if (k_id == dbPoint.get(pI).id && n_id == dbPoint.get(pJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, g - 1).dot(lagrange));
                }
                //l,k
                if (l_id == dbPoint.get(pI).id && k_id == dbPoint.get(pJ).id) {
                    //0
                }
                //l,l
                if (l_id == dbPoint.get(pI).id && l_id == dbPoint.get(pJ).id) {
                    // 0
                }
                //l,m
                if (l_id == dbPoint.get(pI).id && m_id == dbPoint.get(pJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, g - 1).dot(lagrange));
                }
                //l,n
                if (l_id == dbPoint.get(pI).id && n_id == dbPoint.get(pJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, 1 - g).dot(lagrange));
                }
                //m,k
                if (m_id == dbPoint.get(pI).id && k_id == dbPoint.get(pJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, 1 - g).dot(lagrange));
                }
                //m,l
                if (m_id == dbPoint.get(pI).id && l_id == dbPoint.get(pJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, g - 1).dot(lagrange));
                }
                //m,m
                if (m_id == dbPoint.get(pI).id && m_id == dbPoint.get(pJ).id) {
                    //0
                }
                //m,n
                if (m_id == dbPoint.get(pI).id && n_id == dbPoint.get(pJ).id) {
                    // 0
                }
                //n,k
                if (n_id == dbPoint.get(pI).id && k_id == dbPoint.get(pJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, g - 1).dot(lagrange));
                }
                //n,l
                if (n_id == dbPoint.get(pI).id && l_id == dbPoint.get(pJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, 1 - g).dot(lagrange));
                }
                //n,m
                if (n_id == dbPoint.get(pI).id && m_id == dbPoint.get(pJ).id) {
                    // 0
                }
                //n,n
                if (n_id == dbPoint.get(pI).id && n_id == dbPoint.get(pJ).id) {
                    // 0
                }
                j++;
            }
            i++;
        }

        return mt;
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
        return m_id;
    }

    @Override
    public int getN() {
        return n_id;
    }

    @Override
    public int getParametr() {
        return param_id;
    }

    @Override
    public double getNorm() {
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
        MatrixDouble mt = getValue();
        return mt.get(0, 0) / (LK.length() * NM.length());
    }
}
