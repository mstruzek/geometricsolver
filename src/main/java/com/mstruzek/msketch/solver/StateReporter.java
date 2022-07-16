package com.mstruzek.msketch.solver;

public class StateReporter {

    static final boolean DebugEnabled = true;

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
}
