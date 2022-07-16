package com.mstruzek.controller;

public final class Reporter {

    private Reporter() {
    }

    public static void notify(String message) {
        System.out.println(message);
    }

    public static void notify(String format, Object... args) {
        System.out.printf(format, args);
    }

}
