package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Klasa reprezentujaca wiez rownoleglo�ci pomiedzy
 * liniami(vectorami)
 *
 * @author root
 */
public class ConstraintLinesParallelism extends Constraint {


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
     * Vector M - gdy wiez pomiedzy fixline
     */
    Vector m = null;
    /**
     * Point N-id
     */
    int n_id;
    /**
     * Vector N - gdy wiez pomiedzy fixline
     */
    Vector n = null;

    /**
     * macierz rotacji o 90 stopni
     */
    static MatrixDouble R = MatrixDouble.getRotation2x2(90 + 180);
    static MatrixDouble mR = MatrixDouble.getRotation2x2(90); //mR=-R

    /**
     * Konstruktor pomiedzy 4 punktami lub
     * 2 punktami i FixLine(czyli 2 wektory)
     * rownanie tego wiezu to (L-K)x(N-M) = 0
     * iloczyn wektorowy
     * FIXME - zastanowic sie czy nie zrobic abs((L-K)x(N-M)) =0 moze bedzie szybciej zbiegal
     *
     * @param constId
     * @param K
     * @param L
     * @param M
     * @param N
     */
    public ConstraintLinesParallelism(int constId, Point K, Point L, Vector M, Vector N) {
        super(constId, GeometricConstraintType.LinesParallelism, true);

        k_id = K.id;
        l_id = L.id;
        if ((M instanceof Point) && (N instanceof Point)) {
            m_id = ((Point) M).id;
            n_id = ((Point) N).id;
        } else {
            m = M;
            n = N;
        }
    }

    public String toString() {
        MatrixDouble out = getValue();
        double norm = Matrix.constructWithCopy(out.getArray()).norm1();
        if (m == null && n == null)
            return "Constraint-LinesParallelism" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) + "} \n";
        else {
            return "Constraint-LinesParallelism" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,vecM =" + m + ",vecN =" + n + "} \n";
        }

    }

    @Override
    public MatrixDouble getValue() {
        Vector out = new Vector(dbPoint.get(l_id));
        out = out.sub((Vector) dbPoint.get(k_id));
        //out =out.unit();

        MatrixDouble mt = new MatrixDouble(1, 1);

        if ((m == null) && (n == null)) {
            mt.m[0][0] = out.cross(((Vector) dbPoint.get(n_id)).sub((Vector) dbPoint.get(m_id)));
        } else {
            mt.m[0][0] = out.cross(n.sub(m));
        }
        return mt;
    }

    @Override
    public MatrixDouble getJacobian() {
        //macierz 2 wierszowa
        MatrixDouble out = MatrixDouble.fill(1, dbPoint.size() * 2, 0.0);
        //zerujemy cala macierz + wstawiamy na odpowiednie miejsce Jacobian wiezu
        int j = 0;
        if ((m == null) && (n == null)) {
            for (Integer i : dbPoint.keySet()) {

                Point pointI = dbPoint.get(i);

                //a tu wstawiamy macierz dla tego wiezu

                if (k_id == pointI.id) {
                    Vector v1 = ((Vector) dbPoint.get(n_id)).sub((Vector) dbPoint.get(m_id));
                    out.m[0][j * 2] += -v1.y;
                    out.m[0][j * 2 + 1] = v1.x;
                }
                if (l_id == pointI.id) {
                    Vector v1 = ((Vector) dbPoint.get(n_id)).sub((Vector) dbPoint.get(m_id));
                    out.m[0][j * 2] += v1.y;
                    out.m[0][j * 2 + 1] = -v1.x;
                }
                //a tu wstawiamy macierz dla tego wiezu
                if (m_id == pointI.id) {
                    Vector v1 = ((Vector) dbPoint.get(l_id)).sub((Vector) dbPoint.get(k_id));
                    out.m[0][j * 2] += v1.y;
                    out.m[0][j * 2 + 1] = -v1.x;
                }
                if (n_id == pointI.id) {
                    Vector v1 = ((Vector) dbPoint.get(l_id)).sub((Vector) dbPoint.get(k_id));
                    out.m[0][j * 2] += -v1.y;
                    out.m[0][j * 2 + 1] = v1.x;
                }
                j++;
            }
        } else {
            Vector v1 = n.sub(m);
            for (Integer i : dbPoint.keySet()) {

                Point pointI = dbPoint.get(i);

                //a tu wstawiamy macierz dla tego wiezu
                if (k_id == pointI.id) {
                    out.m[0][j * 2] += -v1.y;
                    out.m[0][j * 2 + 1] = v1.x;
                }
                if (l_id == pointI.id) {
                    out.m[0][j * 2] += v1.y;
                    out.m[0][j * 2 + 1] = -v1.x;
                }
                //reszta dla m,n =0
                j++;
            }
        }

        return out;
    }

    @Override
    public boolean isJacobianConstant() {
        if ((m == null) && (n == null)) {
            return false;
        } else {
            //jezeli m,n vectory to constatnt i wtedy Hessian =0
            return true;
        }

    }

    @Override
    public MatrixDouble getHessian(double alfa) {

        //macierz NxN
        MatrixDouble mt = MatrixDouble.fill(dbPoint.size() * 2, dbPoint.size() * 2, 0.0);

        if ((m == null) && (n == null)) {
            //same punkty
            int i = 0;
            for (Integer vI : dbPoint.keySet()) { //wiersz
                int j = 0;
                Point pointI = dbPoint.get(vI);
                for (Integer vJ : dbPoint.keySet()) { //kolumna
                    Point pointJ = dbPoint.get(vJ);
                    //wstawiamy I,-I w odpowiednie miejsca
                    //k,m
                    if (k_id == pointI.id && m_id == pointJ.id) {
                        mt.addSubMatrix(2 * i, 2 * j, R);
                    }
                    //k,n
                    if (k_id == pointI.id && n_id == pointJ.id) {
                        mt.addSubMatrix(2 * i, 2 * j, mR);
                    }
                    //l,m
                    if (l_id == pointI.id && m_id == pointJ.id) {
                        mt.addSubMatrix(2 * i, 2 * j, mR);
                    }
                    //l,n
                    if (l_id == pointI.id && n_id == pointJ.id) {
                        mt.addSubMatrix(2 * i, 2 * j, R);
                    }
                    //m,k
                    if (m_id == pointI.id && k_id == pointJ.id) {
                        mt.addSubMatrix(2 * i, 2 * j, mR);
                    }
                    //m,l
                    if (m_id == pointI.id && l_id == pointJ.id) {
                        mt.addSubMatrix(2 * i, 2 * j, R);
                    }
                    //n,k
                    if (n_id == pointI.id && k_id == pointJ.id) {
                        mt.addSubMatrix(2 * i, 2 * j, R);
                    }
                    //n,l
                    if (n_id == pointI.id && l_id == pointJ.id) {
                        mt.addSubMatrix(2 * i, 2 * j, mR);
                    }
                    j++;
                }
                i++;
            }

            return mt;
        } else {
            // HESSIAN ZERO //m,n - vectory
            return null;
        }
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
        return -1;
    }

    /**
     * @param args
     */
    public static void main(String[] args) {

        Point pn1 = new Point(Point.nextId(), 0.0, 0.0);
        Point pn2 = new Point(Point.nextId(), 10.0, 0.0);
        Point pn3 = new Point(Point.nextId(), 1., 1.0);
        Point pn4 = new Point(Point.nextId(), 10.0, 1.1);

        ConstraintLinesParallelism cn = new ConstraintLinesParallelism(Constraint.nextId(), pn1, pn2, pn3, pn4);
        System.out.println(Constraint.dbConstraint);
        System.out.println(cn.getJacobian());
        System.out.println(cn.getValue());
        System.out.println(cn.getNorm());

    }

    @Override
    public double getNorm() {

        double val = getValue().m[0][0];

        Vector out = new Vector(((Vector) dbPoint.get(k_id)).sub((Vector) dbPoint.get(l_id)));
        val = val / out.length();

        if ((m == null) && (n == null)) {
            val = val / ((Vector) dbPoint.get(m_id)).sub((Vector) dbPoint.get(n_id)).length();
        } else {
            val = val / (m.sub(n)).length();
        }

        return val;
    }
}
