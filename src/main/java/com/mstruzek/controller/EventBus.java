package com.mstruzek.controller;

import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class EventBus{

    private static final Executor executor=Executors.newFixedThreadPool(4);

    private static CopyOnWriteArrayList<Listener> registrations=new CopyOnWriteArrayList<>();

    /**
     * Register Event Handler Function
     *
     * @param eventType subscription type
     * @param function  handler function
     */
    public static void addListener(String eventType,EventHandler function){
        registrations.add(new Listener(eventType,function));
    }

    /**
     * Send event to all subscribed components.
     *
     * @param eventType all types starting with prefix
     * @param arguments additional arguments
     */
    public static void send(String eventType,Object[] arguments){
        for(Listener listener: registrations){
            if(eventType.startsWith(listener.type)){
                executor.execute(()->listener.function.call(eventType,arguments));
            }
        }
    }


    private static class Listener{
        final String type;
        final EventHandler function;
        public Listener(String type,EventHandler function){
            this.type=type;
            this.function=function;
        }

    }


    public interface EventHandler{
        void call(String eventType,Object[] arguments);
    }
}
