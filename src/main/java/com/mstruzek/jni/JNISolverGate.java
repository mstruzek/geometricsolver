package com.mstruzek.jni;

import com.mstruzek.msketch.ConstraintType;
import com.mstruzek.msketch.GeometricType;

public class JNISolverGate {

    
    static final int JNI_SUCCESS = 0;

    static final int JNI_ERROR = 1;


    /**
     * Initialize Nvidia Cuda driver, return 0 success value when all requirements met.
     * 
     * @return error code
     */
    public static native int initDriver();


    public static native java.lang.String getLastError();


    /**
     * Close drive and release all associated resources acquired for computations.
     * 
     * @return error code
     */
    public static native int closeDriver();


    /**
     * Reset computation context. 
     * - release all point registered in context.
     * - release all geometric primitives registered in context.
     * - release all constraints with parameters registered in context.
     * 
     * @return error code
     */
    public static native int resetComputationData();


    /**
     * Reset computation context. 
     * - release all matrices and workspace mem blocks used by Solver factorization methods.
     * 
     * @return error code
     */    
    public static native int resetComputationContext();


    /**
     * Initialize solver context just after stage registration (points,geometrics, and constraints).
     * 
     * @return error code
     */
    public static native int initComputationContext();


    /**
     * Initialize computation context. Prepare all matrices and initialize kernel computations round.
     * 
     * @return error code
     */ 
    public static native int solveSystem();

    


    /**
     * Register point in following computation context.
     * 
     * @param id
     * @param px,py
     * @return error code
     */
     public static native int registerPointType(int id, double px, double py);


    /**
     * Register geometric primitive type in following computation context.
     * 
     * @param id
     * @param p1,p2,p3 control points
     * @param a,b,c,d guide points
     * @return error code
     */
    public static native int registerGeometricType(int id, int geometricTypeId, int p1, int p2, int p3, int a, int b, int c, int d);


    /**
     * Register parameter type in following computation context.
     * 
     * @param id
     * @param value
     * @return error code
     */
    public static native int registerParameterType(int id,  double value);


    /**
     * Register constraint type in following computation context.
     * 
     * @param id
     * @param k,l,m,n
     * @param paramId
     * @param vecX,vecY
     * @return error code
     */
    public static native int registerConstraintType(int id, int constraintTypeId, int k, int l, int m, int n, int paramId, double vecX, double vecY);


    /**
     * Read registered point updated PX coordinates from solver computation round.
     *
     * @param id point 
     * @return px coordinate x value
     */
    public static native double getPointPXCoordinate(int id);


    /**
     * Read registered point updated PY coordinates from solver computation round.
     *
     * @param id point 
     * @return py coordinate y value
     */
    public static native double getPointPYCoordinate(int id);


    /**
     * Return all registerd coordinates (computed) in last processing round. Output Vector Layout:  j - point id ,  double  px = vector[2*j + 0] , double py = vector[2*j + 1]     
     * 
     * @return `JVM registerd global state vector 
     */
    public static native double[] getPointCoordinateVector();

}
