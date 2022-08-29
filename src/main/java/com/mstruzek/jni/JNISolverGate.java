package com.mstruzek.jni;


import com.mstruzek.msketch.solver.SolverStat;

public class JNISolverGate {

    public static final int JNI_SUCCESS = 0;
    public static final int JNI_ERROR = 1;

    static {
        /**
         *
         *   -Djava.library.path=lib
         */
        System.loadLibrary("libgsketcherjni");
    }

    /**
     * Lookup last error from cuda context.
     * @return error descriptor
     */
    public static native java.lang.String getLastError();

    /**
     * Enable/disable setting property.
     * setting id - DEBUG_TENSOR_A , DEBUG_TENSOR_B , DEBUG_TENSOR_SV , CLOCK_MILLISECONDS , CLOCK_NANOSECONDS ,
     * SOLVER_INC_HESSIAN , DEBUG_SOLVER_CONVERGENCE ,
     * @param id
     * @param value
     * @return
     */
    public static native int setBooleanProperty(int id, boolean value);

    /**
     * Update long setting property.
     * setting id - GRID_SIZE , BLOCK_SIZE
     * @param id
     * @param value
     * @return
     */
    public static native int setLongProperty(int id, long value);

    /**
     * Update double setting property.
     * setting id - CU_SOLVER_LWORK_FACTOR
     * @param id
     * @param value
     * @return
     */
    public static native int setDoubleProperty(int id, double value);

    /**
     * Initialize Nvidia Cuda driver, return 0 success value when all requirements met.
     * @param deviceId cuda capable device id
     * @return error code
     */
    public static native int initDriver(int deviceId);

    /**
     * Initialize solver context cudaSteam, cudaEvents, dev_ev, ev.
     * Everything destroyed in destroyComputationContext().
     * - structural data not dependent on f(N)
     * @return error code
     */
    public static native int initComputationContext();

    /**
     * Observe time of last structural changes applied into model (register_ function family).
     * @return 0L or unix timestamp in [ ms ]
     */
    public static native long getCommitTime();

    /**
     * Register point in following computation context.
     * @param id
     * @param px,py
     * @return error code
     */
    public static native int registerPointType(int id, double px, double py);

    /**
     * Register geometric primitive type in following computation context.
     * @param id
     * @param p1,p2,p3 control points
     * @param a,b,c,d  guide points
     * @return error code
     */
    public static native int registerGeometricType(int id, int geometricTypeId, int p1, int p2, int p3, int a, int b, int c, int d);

    /**
     * Register parameter type in following computation context.
     * @param id
     * @param value
     * @return error code
     */
    public static native int registerParameterType(int id, double value);

    /**
     * Register constraint type in following computation context.
     * @param id
     * @param k,l,m,n
     * @param paramId
     * @param vecX,vecY
     * @return error code
     */
    public static native int registerConstraintType(int id, int constraintTypeId, int k, int l, int m, int n, int paramId, double vecX, double vecY);

    /**
     * Initialize solver context just after stage registration (points,geometrics, and constraints).
     *
     * if no structural changes do nothing
     * otherwise free old mem and allocate new context
     * @return error code
     */
    public static native int initComputation();

    /**
     * Initialize computation context. Prepare all matrices [f(N)] and initialize kernel computations round.
     * @return error code
     */
    public static native int solveSystem();

    /**
     * Get last round solver computation statistics.
     * @return solver statistics
     */
    public static native SolverStat getSolverStatistics();

    /**
     * Return all registerd coordinates (computed) in last processing round.
     * Output Vector Layout:  j - point id ,  double  px = vector[2*j + 0] , double py = vector[2*j + 1]
     * @return `JVM registerd global state vector
     */
    public static native double[] fetchStateVector();

    /**
     * Update location of free points after relaxation procedure.
     * @param stateVector []
     * @return
     */
    public static native  int updateStateVector(double[] stateVector);

    /**
     * The only purpose is to set ConstraintFixPoint consts after model relaxation.
     * @param constraintId constraint id
     * @param vecX fixed vector
     * @param vecY fixed vector
     * @return
     */
    public static native int updateConstraintState(int[] constraintId, double[] vecX, double[] vecY, int size);

    /**
     * Update parameter set corresponding values.
     * @param parameterId
     * @param value
     * @param size
     * @return
     */
    public static native int updateParametersValues(int[] parameterId, double[] value, int size);

    /**
     * Read registered point updated PX coordinates from solver computation round.
     * @param id point
     * @return px coordinate x value
     */
    public static native double getPointPXCoordinate(int id);

    /**
     * Read registered point updated PY coordinates from solver computation round.
     * @param id point
     * @return py coordinate y value
     */
    public static native double getPointPYCoordinate(int id);

    /**
     * Destroy computation context after structural change - open file /clear stage.
     * - free all point and geometric primitives registered in context.
     * - free all constraints with parameters registered in context.
     * - free all additional dependent structures on model. ( SV , f(N) )
     *
     * @return error code
     */
    public static native int destroyComputation();

    /**
     * Destroy computation context.
     * - release all matrices and workspace mem blocks used by Solver factorization methods.
     * - free mem for cudaStream_t, cudaEvent_t
     * - free mem of computation structures dev_ev, ev,
     * @return error code
     */
    public static native int destroyComputationContext();

    /**
     * Close drive and release all associated resources acquired for computations.
     * @return error code
     */
    public static native int closeDriver();

}
