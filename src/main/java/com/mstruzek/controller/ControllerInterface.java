package com.mstruzek.controller;

import com.mstruzek.msketch.GeometricConstraintType;
import com.mstruzek.msketch.Vector;

/**
 * Interfejs do oddzialywania na Controler(Model)
 *
 * @author root
 */
public interface ControllerInterface {

    /**
     * Dodaj linie
     */
    void addLine(Vector v1, Vector v2);

    /**
     * Dodaj kolo
     */
    void addCircle(Vector v1, Vector v2);

    /**
     * Dodaj Luk
     */
    void addArc(Vector v1, Vector v2,Vector v3);

    /**
     * Dodaj punkt
     */
    void addPoint(Vector v1);

    /**
     * Rozwiaz system
     */
    void solveSystem();

    /**
     * Przelicz punkty kontrolne -wyzeruj sily
     */
    void evaluateGuidePoints();

    /**
     * Wprowadza zaklocenia na punktach
     */
    void relaxControlPoints(double coefficient);

    /**
     * Dodaj wiez do modelu
     */
    void addConstraint(GeometricConstraintType constraintType, int K, int L, int M, int N, double d);


}
