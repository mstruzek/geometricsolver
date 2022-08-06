package com.mstruzek.msketch.solver;

public class StateReporter {

    private static final StateReporter instance = new StateReporter();

    public static boolean DebugEnabled = false;

    private StateReporter() {}

    public static StateReporter getInstance() {
        return instance;
    }

    public void writelnf(String format, Object... args) {
        System.out.printf(format + "\n", args);
    }

    public void writeln(String message) {
        System.out.println(message);

    }

    public void debug(String message) {
        if(DebugEnabled) {
            System.out.println(message);
        }
    }

    public static boolean isDebugEnabled() {
        return DebugEnabled;
    }

    public void reportSolverStatistics(SolverStat stat) {
        writeln("#=================== Time Space ===================#");
        writelnf("startTime         [ns]: %,20d ", stat.startTime);
        writelnf("stopTime          [ns]: %,20d ", stat.stopTime);
        writelnf("time elapsed      [ns]: %,20d ", stat.stopTime - stat.startTime);
        writelnf("Acc Evaluation    [ns]: %,20d ", stat.accEvaluationTime);
        writelnf("Acc Solver time   [ns]: %,20d ", stat.accSolverTime);
        writeln("");

        writeln("#================== Solver space ==================#");
        writelnf("State vector dimension  : %d ", stat.size);
        writelnf("Coefficients            : %d ", stat.coefficientArity);
        writelnf("Matrix (A) dimension    : %s ", stat.dimension + " x " + stat.dimension);

        writeln("");
        writeln("#================== Error space ==================#");
        writelnf("convergence               : %s" , (stat.convergence ? "T" : "F"));
        writelnf("`error                    : %e" , stat.error);
        writelnf("constraint delta (error)  : %e" , stat.constraintDelta);
        writelnf("iterations  (n)           : %d" , stat.iterations);
        writeln("");
    }
}
