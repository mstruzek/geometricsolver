package com.mstruzek;

import com.mstruzek.jni.JNISolverGate;
import com.mstruzek.msketch.solver.SolverStat;

import java.text.DecimalFormatSymbols;
import java.time.LocalTime;
import java.util.Locale;

import static java.lang.System.out;

public class Main {
    public static void main(String[] args) {

        out.println("Hello world!");


        String DOUBLE_STR_FORMAT = " %11.2e";

        String r = String.format( Locale.ROOT, "test -- " + DOUBLE_STR_FORMAT, 1203.12312);

        out.println(r);


        try {
//            SolverStat solverStatistics = JNISolverGate.getSolverStatistics();
//
//            System.out.printf("accEvaluationTime = %d\n", solverStatistics.accEvaluationTime);
//
//
//*
//             * Test completed - compiled CU kernel with cublas and cusolver in libs !
//
//
//            JNISolverGate.solveSystem();



        } catch (Throwable e) {

            e.printStackTrace();
//            System.err.println("last error" + JNISolverGate.getLastError());

        }




    }
}