package com.mstruzek.msketch.solver;

import java.util.function.Supplier;

public class StateReporter {

    private static final StateReporter instance = new StateReporter();

    static final boolean DebugEnabled = false;

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

    public void debug(Supplier<String> message) {
        if(DebugEnabled) {
            System.out.println(message.get());
        }
    }
}
