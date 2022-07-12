package com.mstruzek.controller;

class ObjectDeserializer {

    public static Integer toInteger(String fieldValue) {
        return Integer.parseInt(fieldValue);
    }

    public static Double toDouble(String fieldValue) {
        return Double.parseDouble(fieldValue);
    }

    public static String toString(String fieldValue) {
        return fieldValue;
    }
}
