package com.mstruzek.controller;

import com.mstruzek.msketch.ConstraintType;
import com.mstruzek.msketch.GeometricType;

import java.util.Locale;


public class ObjectSerializer {

    private static final Locale locale = Locale.ROOT;

    static public String writeToString(int value) {
        return String.format(locale, "%d", value);
    }

    static public String writeToString(double value) {
        return String.format(locale, "%e", value);
    }

    static public String writeToString(ConstraintType constraintType) {
        return constraintType.name();
    }

    public static String writeToString(GeometricType primitiveType) {
        return primitiveType.name();
    }
}
