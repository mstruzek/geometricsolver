package com.mstruzek.controller;

import com.mstruzek.msketch.GeometricConstraintType;
import com.mstruzek.msketch.GeometricPrimitiveType;

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

    public static String writeToString(GeometricPrimitiveType primitiveType) {
        return primitiveType.name();
    }
}
