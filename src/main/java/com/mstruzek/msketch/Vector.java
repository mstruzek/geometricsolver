package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;
import com.mstruzek.msketch.matrix.SmallMatrixDouble;

/**
 * Klasa wektora 2D
 * + podstawowe operacje , iloczyn skalarny i wektorowy , dlugosc ;
 *
 * @author root
 */
public class Vector {

    double x, y;

    public Vector(double x, double y) {
        super();
        this.x = x;
        this.y = y;
    }

    /**
     * Zwraca sume bierzacego i zadanego wektora; C=A+B
     */
    public Vector add(Vector vec1) {
        return new Vector(this.x + vec1.x, this.y + vec1.y);
    }

    public Vector add(double d) {
        return new Vector(this.x + d, this.y + d);
    }

    public String toString() {
        return "[" + this.x + " , " + this.y + "] ";
    }


    /**
     * Zwraca roznice aktualnego wektora i zadanego ;COPY : C=A-B;
     */
    public Vector sub(Vector vec1) {
        return new Vector(this.x - vec1.x, this.y - vec1.y);
    }

    /**
     * Zwraca mno�enie wektora przez scalar C=alfa*A;
     */
    public Vector dot(double scalar) {
        return new Vector(this.x * scalar, this.y * scalar);
    }

    /**
     * Zwraca iloczyn skalarny aktualnego i zadanego wektora; C=A'*B;
     */
    public double dot(Vector vec1) {
        return (this.x * vec1.x + this.y * vec1.y);
    }


    /**
     * Zwraca iloczyn wektorowy aktualnego i zadanego wektora; C =A x B ;
     */
    public double cr(Vector vec1) {
        return (this.x * vec1.y - this.y * vec1.x);
    }

    /**
     * Equivalent of rotation vector by MatrixDouble of R*a = ~a , R  = [ 0 -1 , 1  0]
     *
     * @return
     */
    public Vector cr() {
        return new Vector(-this.y, this.x);
    }

    /**
     * Zwraca dlugosc wektora
     */
    public double length() {
        return Math.sqrt(this.x * this.x + this.y * this.y);
    }


    /**
     * Zwraca wartosc X wektora
     */
    public double getX() {
        return x;
    }

    /**
     * Ustawia wartosc X wektora
     */
    public void setX(double x) {
        this.x = x;
    }

    /**
     * Zwraca wartosc Y wektora
     */
    public double getY() {
        return y;
    }

    /**
     * Ustawia wartosc Y wektora
     */
    public void setY(double y) {
        this.y = y;
    }

    /**
     * Ustawia wartpsco x i y wektora
     */
    public void setLocation(double x, double y) {
        this.x = x;
        this.y = y;
    }

    /**
     * Obruc wektor o dany kat
     *
     * @param angle kat w stopniach
     * @return
     */
    public Vector Rot(double angle) {
        angle = Math.toRadians(angle);
        return new Vector(this.x * Math.cos(angle) - this.y * Math.sin(angle), this.x * Math.sin(angle) + this.y * Math.cos(angle));
    }

    /**
     * Zwraca nowy wektor , ktory jest wektorem jednostkowym
     * czyli wersorem
     *
     * @return new Vector() - copy
     */
    public Vector unit() {
        return new Vector(this.x / this.length(), this.y / this.length());
    }

    public MatrixDouble cartesian(Vector rhs) {
        double a00 = this.x * rhs.x;
        double a01 = this.x * rhs.y;
        double a10 = this.y * rhs.x;
        double a11 = this.y * rhs.y;
        MatrixDouble sm = MatrixDouble.smallMatrix(a00, a01, a10, a11);
        return sm;
    }
}
