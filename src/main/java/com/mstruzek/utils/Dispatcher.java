package com.mstruzek.utils;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

public class Dispatcher {

    private final ExecutorService systemExecutor;

    public Dispatcher(ExecutorService systemExecutor) {
        this.systemExecutor = systemExecutor;
    }

    public void shutdown() {
        systemExecutor.shutdown();
    }

    public List<Runnable> shutdownNow() {
        return systemExecutor.shutdownNow();
    }

    public boolean isShutdown() {
        return systemExecutor.isShutdown();
    }

    public boolean isTerminated() {
        return systemExecutor.isTerminated();
    }

    public boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException {
        return systemExecutor.awaitTermination(timeout, unit);
    }

    public Future<?> submit(Runnable task) {
        return systemExecutor.submit(task);
    }
}
