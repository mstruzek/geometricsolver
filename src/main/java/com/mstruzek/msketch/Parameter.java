package com.mstruzek.msketch;

/**
 * Klasa reprezentuje Parametr - dlugosc lub wartosc kat (w stopniach)
 *
 * @author root
 */
public class Parameter {

    int id;

    /*** parametr */
    double value;

    /**
     * Glowny Konstruktor
     *
     * @param par parametr
     */
    public Parameter(int id, double par) {
        this.id = id;
        this.value = par;
    }

    /**
     * Pobierz parametr
     *
     * @return
     */
    public double getValue() {
        return value;

    }

    public double getRadians() {
        return (Math.PI / 180) * value;
    }

    /**
     * Ustaw/zmien parametr
     *
     * @param param
     */
    public void setValue(double param) {
        this.value = param;
    }

    public String toString() {
        return "param" + id + " = " + value + "\n";

    }

    public int getId() {
        return id;
    }

}
