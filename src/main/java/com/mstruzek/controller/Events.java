package com.mstruzek.controller;

import javax.swing.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


/**
 * Swing Event Bus
 */
public class Events {

    public static boolean DISABLE = false;

    private static CopyOnWriteArrayList<Listener> registrations = new CopyOnWriteArrayList<>();

    public static final int EVENT_DISPATCHER_THREADS = 1;
    private static final ExecutorService eventDispatcher = Executors.newFixedThreadPool(EVENT_DISPATCHER_THREADS);

    /**
     * Register Event Handler Function
     * @param eventType subscription type
     * @param function  handler function
     */
    public static void addListener(String eventType, EventHandler function) {
        registrations.add(new Listener(eventType, function, false));
    }

    public static void addAwtListener(String eventType, EventHandler function) {
        registrations.add(new Listener(eventType, function, true));
    }

    public static void send(String eventType, Object[] arguments) {
        if (DISABLE) {
            return;
        }

        for (Listener l : registrations) {

            if (!eventType.startsWith(l.type))
                continue;

            eventDispatcher.submit(new Runnable() {
                @Override
                public void run() {
                    if(l.isAwt) {
                        SwingUtilities.invokeLater(() -> l.function.call(eventType, arguments));
                    } else {
                        l.function.call(eventType, arguments);
                    }
                }
            });
        }
    }

    private static class Listener {
        final String type;
        final EventHandler function;
        final boolean isAwt;

        public Listener(String type, EventHandler function, boolean isAwt) {
            this.type = type;
            this.function = function;
            this.isAwt = isAwt;
        }
    }

    public interface EventHandler {
        void call(String eventType, Object[] arguments);
    }
}
