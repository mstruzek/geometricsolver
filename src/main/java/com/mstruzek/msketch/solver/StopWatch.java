package com.mstruzek.msketch.solver;

import java.util.function.LongSupplier;

public class StopWatch {
    private static final LongSupplier nanoClock = System::nanoTime;
    long startTick;
    long stopTick;
    long accTime;

    StopWatch() {
    }

    public void startTick() {
        this.startTick = nanoClock.getAsLong();
    }

    public void stopTick() {
        this.stopTick = nanoClock.getAsLong();
        this.accTime += (stopTick - startTick);
    }

    public long delta() {
        return stopTick - startTick;
    }
}
