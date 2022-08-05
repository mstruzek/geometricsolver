package com.mstruzek.controller;

import com.mstruzek.msketch.GeometricConstraintType;
import com.mstruzek.msketch.GeometricType;

public class ObjectSerializer {

    static public String writeToString(int value) {
        return Integer.toString(value);
    }

    static public String writeToString(double value) {
        return Double.toString(value);
    }

    static public String writeToString(GeometricConstraintType constraintType) {
        return constraintType.name();
    }

    public static String writeToString(GeometricType primitiveType) {
        return primitiveType.name();
    }
}
