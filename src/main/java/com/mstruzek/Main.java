package com.mstruzek;

import com.mstruzek.jni.JNISolverGate;
import com.mstruzek.msketch.solver.SolverStat;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello world!");


        try {
            SolverStat solverStatistics = JNISolverGate.getSolverStatistics();

            System.out.printf("accEvaluationTime = %d\n", solverStatistics.accEvaluationTime);


            /**
             * Test completed - compiled CU kernel with cublas and cusolver in libs !
             */
            JNISolverGate.solveSystem();



        } catch (Throwable e) {

            e.printStackTrace();
//            System.err.println("last error" + JNISolverGate.getLastError());

        }




    }
}