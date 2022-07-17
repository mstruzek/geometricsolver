package com.mstruzek.msketch.matrix;

import com.mstruzek.msketch.Point;

/**
 * Utility class bedzie nam wiazala wartosci w danej macierzy z odpowiednimi  wartosciami w punktach
 *
 * @author root
 */
public class PointUtility {

    /**
     * Funkcja przepisuje wartosci z macierzy do wszystkich punktow
     * @param stateVector state vector
     */
    public static void copyFromStateVector(MatrixDouble stateVector) {
        int k = 0;
        for (Integer i : Point.dbPoint.keySet()) {
            double x = stateVector.getQuick(k * 2, 0);
            double y = stateVector.getQuick(k * 2 + 1, 0);
            Point.dbPoint.get(i).setLocation(x, y);
            k++;
        }
    }

    /**
     * Funkcja przepisuje wartosci z punktow do aktualnej macierzy
     * @param stateVector vector vector
     */
    public static void copyIntoStateVector(MatrixDouble stateVector) {
        int k = 0;
        for (Integer i : Point.dbPoint.keySet()) {
            double pointX = Point.dbPoint.get(i).getX();
            double pointY = Point.dbPoint.get(i).getY();
            stateVector.setQuick(k * 2, 0, pointX);
            stateVector.setQuick(k * 2 + 1, 0, pointY);
            k++;
        }
    }

    /**
     * Inicjalizacji wspolczynnikow Lagrange'a
     * @param stateVector state vector
     */
    public static void setupLagrangeMultipliers(MatrixDouble stateVector) {
        double defaultValue = 0.0;
        int size = Point.dbPoint.size() * 2;
        for(int i = size; i < stateVector.height(); i++) {
            stateVector.setQuick(i, 0 , defaultValue);
        }
    }
}
