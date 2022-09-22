package com.mstruzek.utils;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class Dispatchers {

    private static final ThreadFactory DISPATCHER_THREAD_FACTORY = new DispatcherThreadFactory();

    /**
     * New single thread dispatcher. Thread-safe communication channel with controller.
     * @return
     */
    public static Dispatcher newDispatcher() {
        final LinkedBlockingQueue<Runnable> blockingQueue = new LinkedBlockingQueue<>();
        /// single thread pool
        final int corePoolSize = 1;
        /// with upper bound
        final int maxPoolSize = 1;
        final long keepAliveTime = 0L;
        final ThreadPoolExecutor singleThreadPool = new ThreadPoolExecutor(corePoolSize, maxPoolSize, keepAliveTime, TimeUnit.MILLISECONDS,
            blockingQueue, DISPATCHER_THREAD_FACTORY);

        Dispatcher scheduler = new Dispatcher(singleThreadPool);

        return scheduler;
    }

    private static class DispatcherThreadFactory implements ThreadFactory {

        private final ThreadGroup group;
        private final AtomicInteger threadNumber = new AtomicInteger(1);
        private final String namePrefix;

        DispatcherThreadFactory() {
            group = Thread.currentThread().getThreadGroup();
            namePrefix = "eventDispatcher-";
        }

        public Thread newThread(Runnable r) {
            Thread t = new Thread(group, r, namePrefix + threadNumber.getAndIncrement(), 0);
            if (t.isDaemon())
                t.setDaemon(false);
            if (t.getPriority() != Thread.NORM_PRIORITY)
                t.setPriority(Thread.NORM_PRIORITY);
            return t;
        }
    }

}
