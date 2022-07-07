package com.mstruzek.controller;

public class Log{

    private static Log logger=new Log();

    /// FIXME scheduler single thread dispatch to display thread -> textArea

    public static void write(String message){
        System.out.println(message);
    }

    public static void write(String format,Object... args){
        System.out.printf(format,args);
    }

    public static Log getInstance(){
        return logger;
    }
}
