package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import java.util.TreeMap;

/**
 * Klasa reprezentujaca wiez rownej dlugosci pomiedzy
 * dwoma wektorami (4 punkty)
 *
 * @author root
 */
public class ConstraintLinesSameLength extends Constraint{

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
     * Konstruktor pomiedzy 4 punktami lub
     * 2 punktami i FixLine(czyli 2 wektory)
     * rownanie tego wiezu to sqrt[(K-L)'*(K-L)] - sqrt[(M-N)'*(M-N)] = 0
     * iloczyn skalarny
     *
     * @param constId
     * @param K
     * @param L
     * @param M
     * @param N
     */
    public ConstraintLinesSameLength(int constId,Point K,Point L,Point M,Point N){
        this(constId,K,L,M,N,true);
    }

    public ConstraintLinesSameLength(int constId,Point K,Point L,Point M,Point N,boolean storage){
        super(constId,GeometricConstraintType.LinesSameLength,storage);
        k_id=K.id;
        l_id=L.id;
        m_id=M.id;
        n_id=N.id;
    }

    public String toString(){
        MatrixDouble out=getValue(Point.dbPoint,Parameter.dbParameter);
        double norm=Matrix.constructWithCopy(out.getArray()).norm1();
        return "Constraint-LinesSameLength"+constraintId+"*s"+size()+" = "+norm+" { K ="+Point.dbPoint.get(k_id)+"  ,L ="+Point.dbPoint.get(l_id)+" ,M ="+Point.dbPoint.get(m_id)+",N ="+Point.dbPoint.get(n_id)+"} \n";

    }

    @Override
    public MatrixDouble getJacobian(TreeMap<Integer,Point> dbPoints,TreeMap<Integer,Parameter> dbParameter){
        //macierz 2 wierszowa
        MatrixDouble out=MatrixDouble.fill(1,dbPoints.size()*2,0.0);
        //zerujemy cala macierz + wstawiamy na odpowiednie miejsce Jacobian wiezu
        int j=0;
        Vector vLK=((Vector) dbPoints.get(l_id)).sub((Vector) dbPoints.get(k_id)).unit();
        Vector vNM=((Vector) dbPoints.get(n_id)).sub((Vector) dbPoints.get(m_id)).unit();
        for(Integer i: dbPoints.keySet()){

            //a tu wstawiamy macierz dla tego wiezu
            if(k_id==dbPoints.get(i).id){
                out.m[0][j*2]=-vLK.x;
                out.m[0][j*2+1]=-vLK.y;
            }
            if(l_id==dbPoints.get(i).id){
                out.m[0][j*2]=vLK.x;
                out.m[0][j*2+1]=vLK.y;
            }
            //a tu wstawiamy macierz dla tego wiezu
            if(m_id==dbPoints.get(i).id){
                out.m[0][j*2]=vNM.x;
                out.m[0][j*2+1]=vNM.y;
            }
            if(n_id==dbPoints.get(i).id){
                out.m[0][j*2]=-vNM.x;
                out.m[0][j*2+1]=-vNM.y;
            }
            j++;
        }

        return out;
    }

    @Override
    public boolean isJacobianConstant(){
        return true;

    }

    @Override
    public MatrixDouble getValue(TreeMap<Integer,Point> dbPoints,TreeMap<Integer,Parameter> dbParameter){

        Double vLK=((Vector) dbPoints.get(l_id)).sub((Vector) dbPoints.get(k_id)).length();
        Double vNM=((Vector) dbPoints.get(n_id)).sub((Vector) dbPoints.get(m_id)).length();
        MatrixDouble mt=new MatrixDouble(1,1);
        mt.m[0][0]=vLK-vNM;
        return mt;
    }

    @Override
    public MatrixDouble getHessian(TreeMap<Integer,Point> dbPoints,TreeMap<Integer,Parameter> dbParameter){

        //macierz NxN
        MatrixDouble out=MatrixDouble.fill(dbPoints.size()*2,dbPoints.size()*2,0.0);

        return out;
    }

    @Override
    public boolean isHessianConstant(){
        return true;
    }

    @Override
    public int getK(){
        return k_id;
    }

    @Override
    public int getL(){
        return l_id;
    }

    @Override
    public int getM(){
        return m_id;
    }

    @Override
    public int getN(){
        return n_id;
    }

    @Override
    public int getParametr(){
        return -1;
    }

    /**
     * @param args
     */
    public static void main(String[] args){


    }

    @Override
    public double getNorm(TreeMap<Integer,Point> dbPoints,TreeMap<Integer,Parameter> dbParameter){

        return getValue(dbPoints,dbParameter).m[0][0];
    }

}
