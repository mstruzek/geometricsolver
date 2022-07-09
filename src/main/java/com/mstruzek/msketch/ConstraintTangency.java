package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Wiez odpowiedzialny za stycznosc Okregu do Lini
 *
 * @author root
 */
public class ConstraintTangency extends Constraint{

    /** Punkty kontrolne */

    /*** Point K-id */
    int k_id;
    /*** Point L-id */
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
     * macierz rotacji o 90 stopni
     */
    static MatrixDouble R=MatrixDouble.getRotation2x2(90+180);
    static MatrixDouble mR=MatrixDouble.getRotation2x2(90); //mR=-R

    /**
     * Konstruktor pomiedzy 4 punktami lub
     * rownanie tego wiezu to [(L-K)x(M-K)]^2 - (L-K)'*(L-K)*(N-M)'*(N-M) = 0
     *
     * @param constId
     * @param K       punkt prostej
     * @param L       punkt prostej
     * @param M       srodek okregu = p1
     * @param N       promien okregu = p2
     */
    public ConstraintTangency(int constId,Point K,Point L,Point M,Point N){
        super(constId,GeometricConstraintType.Tangency, true);
        k_id=K.id;
        l_id=L.id;
        m_id=M.id;
        n_id=N.id;
    }

    public String toString(){
        MatrixDouble out=getValue();
        double norm=Matrix.constructWithCopy(out.getArray()).norm1();
        return "Constraint-Tangency"+constraintId+"*s"+size()+" = "+norm+" { K ="+ dbPoint.get(k_id)+"  ,L ="+ dbPoint.get(l_id)+" ,M ="+ dbPoint.get(m_id)+",N ="+ dbPoint.get(n_id)+"} \n";
    }

    @Override
    public MatrixDouble getValue(){

        Vector vMK=((Vector) dbPoint.get(m_id)).sub((Vector) dbPoint.get(k_id));
        Vector vLK=((Vector) dbPoint.get(l_id)).sub((Vector) dbPoint.get(k_id));
        Vector vNM=((Vector) dbPoint.get(n_id)).sub((Vector) dbPoint.get(m_id));

        MatrixDouble mt=new MatrixDouble(1,1);

        mt.m[0][0]=Math.sqrt(vLK.cross(vMK)*vLK.cross(vMK))-vLK.length()*vNM.length();

        return mt;
    }

    @Override
    public MatrixDouble getJacobian(){
        //macierz 2 wierszowa
        MatrixDouble out=MatrixDouble.fill(1,dbPoint.size()*2,0.0);
        //zerujemy cala macierz + wstawiamy na odpowiednie miejsce Jacobian wiezu
        int j=0;
        Vector vMK=((Vector) dbPoint.get(m_id)).sub((Vector) dbPoint.get(k_id));
        Vector vLK=((Vector) dbPoint.get(l_id)).sub((Vector) dbPoint.get(k_id));
        Vector vLM=((Vector) dbPoint.get(l_id)).sub((Vector) dbPoint.get(m_id));
        Vector vNM=((Vector) dbPoint.get(n_id)).sub((Vector) dbPoint.get(m_id));
        double g=vLK.cross(vMK);
        double zn=Math.signum(g);

        for(Integer i: dbPoint.keySet()){

            //a tu wstawiamy macierz dla tego wiezu
            if(k_id==dbPoint.get(i).id){
                out.m[0][j*2]=zn*vLM.y+vNM.length()*vLK.unit().x;
                out.m[0][j*2+1]=-zn*vLM.x+vNM.length()*vLK.unit().y;
            }
            if(l_id==dbPoint.get(i).id){
                out.m[0][j*2]=zn*vMK.y-vNM.length()*vLK.unit().x;
                out.m[0][j*2+1]=-zn*vMK.x-vNM.length()*vLK.unit().y;
            }
            //a tu wstawiamy macierz dla tego wiezu
            if(m_id==dbPoint.get(i).id){
                out.m[0][j*2]=-zn*vLK.y+vLK.length()*vNM.unit().x;
                out.m[0][j*2+1]=zn*vLK.x+vLK.length()*vNM.unit().y;
            }
            if(n_id==dbPoint.get(i).id){
                out.m[0][j*2]=-vLK.length()*vNM.unit().x;
                out.m[0][j*2+1]=-vLK.length()*vNM.unit().y;
            }
            j++;
        }
        return out;
    }

    @Override
    public boolean isJacobianConstant(){
        return false;
    }

    @Override
    public MatrixDouble getHessian(){

        //macierz NxN
        MatrixDouble out=MatrixDouble.fill(dbPoint.size()*2,dbPoint.size()*2,0.0);

        Vector vMK=((Vector) dbPoint.get(m_id)).sub((Vector) dbPoint.get(k_id));
        Vector vLK=((Vector) dbPoint.get(l_id)).sub((Vector) dbPoint.get(k_id));
        Vector vLM=((Vector) dbPoint.get(l_id)).sub((Vector) dbPoint.get(m_id));
        Vector vNM=((Vector) dbPoint.get(n_id)).sub((Vector) dbPoint.get(m_id));

        double g=vLK.cross(vMK);
        double zn=Math.signum(g); //znak
        double k=vNM.unit().dot(vLK.unit());

        int i=0;
        //same punkty
        for(Integer vI: dbPoint.keySet()){ //wiersz
            int j=0;
            for(Integer vJ: dbPoint.keySet()){ //kolumna
                //k,k
                if(k_id==dbPoint.get(vI).id && k_id==dbPoint.get(vJ).id){
                    // 0
                }
                //k,l
                if(k_id==dbPoint.get(vI).id && l_id==dbPoint.get(vJ).id){
                    out.setSubMatrix(2*i,2*j,R.dotC(zn));
                }
                //k,m
                if(k_id==dbPoint.get(vI).id && m_id==dbPoint.get(vJ).id){
                    out.setSubMatrix(2*i,2*j,mR.dotC(zn).addC(MatrixDouble.diagonal(2,-k)));
                }
                //k,n
                if(k_id==dbPoint.get(vI).id && n_id==dbPoint.get(vJ).id){
                    out.setSubMatrix(2*i,2*j,MatrixDouble.diagonal(2,k));
                }
                //l,k
                if(l_id==dbPoint.get(vI).id && k_id==dbPoint.get(vJ).id){
                    out.setSubMatrix(2*i,2*j,mR.dotC(zn));
                }
                //l,l
                if(l_id==dbPoint.get(vI).id && l_id==dbPoint.get(vJ).id){
                    // 0
                }
                //l,m
                if(l_id==dbPoint.get(vI).id && m_id==dbPoint.get(vJ).id){
                    out.setSubMatrix(2*i,2*j,R.dotC(zn).addC(MatrixDouble.diagonal(2,k)));
                }
                //l,n
                if(l_id==dbPoint.get(vI).id && n_id==dbPoint.get(vJ).id){
                    out.setSubMatrix(2*i,2*j,MatrixDouble.diagonal(2,-k));
                }
                //m,k
                if(m_id==dbPoint.get(vI).id && k_id==dbPoint.get(vJ).id){
                    out.setSubMatrix(2*i,2*j,R.dotC(zn).addC(MatrixDouble.diagonal(2,-k)));
                }
                //m,l
                if(m_id==dbPoint.get(vI).id && l_id==dbPoint.get(vJ).id){
                    out.setSubMatrix(2*i,2*j,mR.dotC(zn).addC(MatrixDouble.diagonal(2,k)));
                }
                //m,m
                if(m_id==dbPoint.get(vI).id && m_id==dbPoint.get(vJ).id){
                    //0
                }
                //m,n
                if(m_id==dbPoint.get(vI).id && n_id==dbPoint.get(vJ).id){
                    // 0
                }
                //n,k
                if(n_id==dbPoint.get(vI).id && k_id==dbPoint.get(vJ).id){
                    out.setSubMatrix(2*i,2*j,MatrixDouble.diagonal(2,k));
                }
                //n,l
                if(n_id==dbPoint.get(vI).id && l_id==dbPoint.get(vJ).id){
                    out.setSubMatrix(2*i,2*j,MatrixDouble.diagonal(2,-k));
                }
                //n,m
                if(n_id==dbPoint.get(vI).id && m_id==dbPoint.get(vJ).id){
                    // 0
                }
                //n,n
                if(n_id==dbPoint.get(vI).id && n_id==dbPoint.get(vJ).id){
                    // 0
                }
                j++;
            }
            i++;
        }

        return out;
    }

    @Override
    public boolean isHessianConstant(){
        return false;
    }

    /**
     * @param args
     */
    public static void main(String[] args){
        Point pn1=new Point(Point.nextId(),0.0,0.0);
        Point pn2=new Point(Point.nextId(),10.0,0.01);
        Point pn3=new Point(Point.nextId(),0.01,10.0);
        Point pn4=new Point(Point.nextId(),10.0,10.0);
        //Vector pn3 = new Vector(1.0,1.0);
        //Vector pn4 = new Vector(1.0,2.0);
        //System.out.println(ConstraintTangency.R);
        ConstraintTangency cn=new ConstraintTangency(Constraint.nextId(),pn1,pn2,pn3,pn4);
        System.out.println(Constraint.dbConstraint);
        System.out.println(cn.getJacobian());
        System.out.println(cn.getValue());

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

    @Override
    public double getNorm(){
        return 0;
    }
}
