package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.TensorDouble;

import java.util.Arrays;
import java.util.Set;

/**
 * Klasa abstrakcyjna dla podstawowych elementow geometrycznych jakie moga zostac
 * narysowane na ekranie : linia, luk, okrag,"wolny" punkt
 */
public abstract class GeometricObject {

    /**
     * id danego elemntu podstawowego
     */
    protected int primitiveId;
    /**
     * Typ elementu
     */
    protected GeometricType type = null;


    /**
     * tablica przechowujaca powiazane wiezy dla punktow kontrolnych np : a,b,c
     */
    protected Constraint[] constraints = new Constraint[0];

    public GeometricObject(int primitiveId, GeometricType geometricType) {
        this.primitiveId = primitiveId;
        this.type = geometricType;
    }

    /**
     * Przelicza na nowo pozycje punktow prowadzacych
     */
    public abstract void evaluateGuidePoints();


    /**
     * Funkcja zwraca wartosc sil w sprezynach dla poszczegolnych punkt�w w danym {@link GeometricObject}
     */
    public abstract void evaluateForceIntensity(int row, TensorDouble dest);

    /**
     * Funkcja do wyliczenia macierzy sztywnosci elementu - macierz szytnowsci Fq
     * @param row  row offset
     * @param col  column offset
     * @param dest destination matrix
     * @return
     */
    public abstract void setStiffnessMatrix(int row, int col, TensorDouble dest);

    /**
     * Pobierz wszystkie punkty powiazane z dana figura
     */
    public abstract int[] getAllPointsId();

    public abstract Point[] getAllPoints();

    /**
     * Funkcja  ustawia wiezy poczatkowe -czyli rejestruje w bazie wiezow, np : wiez FixPoint na Point[a,b,c]
     */
    public abstract void setAssociateConstraints(Set<Integer> skipIds);


    /**
     * Funkcja zwraca ilosc punktow w danym elemencie geometrycznym
     */
    public abstract int getNumOfPoints();

    /**
     * Funkcja zwraca punktu p1,p2,p3 - potrzebne do wyswietlania
     */
    public abstract int getP1();

    public abstract int getP2();

    public abstract int getP3();

    public abstract int getA();

    public abstract int getB();

    public abstract int getC();

    public abstract int getD();

    /**
     * Zwraca id figury
     * @return
     */
    public int getPrimitiveId() {
        return primitiveId;
    }

    /**
     * Zwraca rodzaj figury
     * @return
     */
    public GeometricType getType() {
        return type;
    }

    /**
     * Funkcja zwraca pelny jakobian sil dla wszystkich elementow geometrycznych
     * @param mt
     */
    public static void evaluateStiffnessMatrix(TensorDouble mt) {
        int rowCol = 0;
        for (GeometricObject geometricObject : ModelRegistry.dbPrimitives.values()) {
            geometricObject.setStiffnessMatrix(rowCol, rowCol, mt);
            rowCol += geometricObject.getNumOfPoints() * 2;
        }
    }

    /**
     * Funkcja zwraca wartosc sil w sprezynach dla wszystkich punktow
     * @param dest destination matrix
     */
    public static void evaluateForceVector(TensorDouble dest) {
        int row = 0;
        for (GeometricObject geometricObject : ModelRegistry.dbPrimitives().values()) {
            geometricObject.evaluateForceIntensity(row, dest);
            row += geometricObject.getNumOfPoints() * 2;
        }
    }

    public Constraint[] associatedConstraints() {
        return constraints;
    }

    public BoundingBox getBoundingBox() {
        return Arrays.stream(getAllPoints()).reduce(new BoundingBox(), BoundingBox::fillInPoint,
            BoundingBox::fillInBoundingBox);
    }

}
