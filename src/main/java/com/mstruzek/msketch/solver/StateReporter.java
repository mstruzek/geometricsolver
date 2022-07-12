package com.mstruzek.msketch.solver;

public class StateReporter {

    public void writelnf(String format, Object... args) {
        System.out.printf(format + "\n", args);
    }

    public void writeln(String message) {
        System.out.println(message);

    }
}
