package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

import java.util.Set;
import java.util.TreeMap;

/*
 * Wiez jedowymairowy,iloczyn skalarny,
 */
public abstract class Constraint implements ConstraintInterface {

    /**
     * Licznik wiezow
     */
    public static int constraintCounter = 0;

    /**
     * numer kolejno utworzonej lini
     */
    protected int constraintId;

    protected GeometricConstraintType constraintType = null;

    /**
     * [ false ] - this constraint is normally not visible unless CTRL function applied into layout
     */
    protected boolean persistent;

    /**
     * tablica wszystkich linii
     */
    public static TreeMap<Integer, Constraint> dbConstraint = new TreeMap<>();

    /*** przesuniecia absolutne punktow wzgledem ukladuw wspolrzednych vectora stanu */
    protected PointLocation po = PointLocation.getInstance();

    public Constraint(Integer constraintId, GeometricConstraintType constraintType, boolean persistent) {
        super();
        this.constraintId = constraintId;
        this.constraintType = constraintType;
        this.persistent = persistent;
        dbConstraint.put(constraintId, this);
    }

    public static Integer nextId() {
        return constraintCounter++;
    }

    public static Integer nextId(Set<Integer> skipIdentifiers) {
        int nextId = constraintCounter++;
        while (skipIdentifiers.contains(nextId)) {
            nextId = constraintCounter++;
        }
        return nextId;
    }


    /**
     * Funkcja zwraca WARTOSC WIEZU w postaci skalaru ,przypadek dla Constraint.size=1
     * lub Vectora kolumnowego gdy size=2;
     * bedziemy uzywac podczas sprawdzania czy wiez jest wyzerowany
     *
     * @return macierz albo 1x1 albo 2x1;
     */
    public abstract MatrixDouble getValue();

    /**
     * Funkcja zwraca Jacobian w postaci wektora wierszowego gdy Constraint.size=1,
     * lub dwa wektory wierszowe gdy size=2;
     *
     * @param mts write computed jacobian into matrix span.
     */
    public abstract void getJacobian(MatrixDouble mts);

    /**
     * Funkcja zwraca norme danego wiezu
     *
     * @return
     */
    public abstract double getNorm();

    /**
     * Funkcja zwraca true jesli Jacobian jest staly
     * ,jezeli tak jest to Hessian = 0
     *
     * @return
     */
    public abstract boolean isJacobianConstant();

    /**
     * Funkcja zwraca Hessian  - macierz (n-points)^2
     * czyli drugie pochodne, ale tylko dla wiezow o size =1
     * Jedyny  wiez o size=2 czyli ConstraintConnect2Point ma staly
     * Jacobian zatem nie ma Hessianu
     */
    public abstract MatrixDouble getHessian(double lagrange);

    /**
     * Funkcja zwraca true jesli Hessian jest staly
     *
     * @return
     */
    public abstract boolean isHessianConst();

    /**
     * Podaje nam ile mnoznikow lagrange'a posiada dany wiez
     * 1 lub 2 co rownowazne jest temu ile wierszy bedzie mial wynik z funkcji
     * getJacobian lub getValue
     *
     * @return
     */
    public int size() {
        return this.constraintType.size();
    }
    // FIXME - funkcja odpowiedzialna za odswiezenie danych w wezle np vectory w FixPoint
    //public abstract void update();

    /**
     * Zwraca calkowit� ilosc mnoznikow lagrange'a od wszystkich wiezow
     *
     * @return
     */
    public static int allLagrangeCoffSize() {
        int coffSize = 0;
        for (Constraint constraint : dbConstraint.values()) {
            coffSize += constraint.size();
        }
        return coffSize;
    }


    /**
     * Funkcja zwraca Jakobian ze wszystkich wiezow
     *
     * @param mt
     * @return macierz ,jakobian wiezow d(Wiezy)/dq
     */
    public static void getFullJacobian(MatrixDouble mt) {
        int rowPos = 0;

        for (Integer id : dbConstraint.keySet()) {
            Constraint.dbConstraint.get(id).getJacobian(mt.viewSpan(rowPos, 0, Constraint.dbConstraint.get(id).size(), mt.width()));
            rowPos += Constraint.dbConstraint.get(id).size();
        }
    }

    /**
     * Funkcja zwraca prawe strony , czyli wartosci wszystkich wiezow
     * w wektorze kolumnowym
     *
     * @param mt
     * @return
     */
    public static void getFullConstraintValues(MatrixDouble mt) {
        int currentRow = 0;
        for (Integer id : dbConstraint.keySet()) {
            mt.setSubMatrix(currentRow, 0, Constraint.dbConstraint.get(id).getValue());
            currentRow += Constraint.dbConstraint.get(id).size();
        }
    }

    /**
     * Funkcja zwraca calkowita norme dla wszystkich wiezow
     *
     * @return
     */
    public static double getFullNorm() {
        double norm = 0;
        for (Integer id : dbConstraint.keySet()) {
            double consNorm = Constraint.dbConstraint.get(id).getNorm();
            norm += (consNorm * consNorm);
        }
        return Math.sqrt(norm);
    }

    /**
     * Funkcja zwraca macierz hessianu dla wszystkich wiezow d(Jak'*a)/dq * ( gdzie  a -  Lagrange coefficient )
     *
     * @param hs   Full  HESSIAN
     * @param sv   wektor stanu x i w dolnej czesci wektor z  mnoznikami lagrange'a
     * @param size liczebnosc prymitywnych punktow
     * @return
     */
    public static MatrixDouble getFullHessian(MatrixDouble hs, MatrixDouble sv, int size) {

        int offset;             //licznik mnoznikow lagrange'a
        double lagrange;        //wartosc aktualnego mnoznika
        MatrixDouble conHs;

        offset = 0;
        for (Integer id : dbConstraint.keySet()) {
            if (!Constraint.dbConstraint.get(id).isJacobianConstant()) {
                /// pierwsza pochodna po wektorze stanu jest nie const, tak wiec Hessian do przeliczenia

                lagrange = sv.getQuick(size + offset, 0);

                ///
                ///   Hessian - dla tego wiezu liczony na cala macierz !
                ///     TODO  !!!!   langrange ===????

                conHs = Constraint.dbConstraint.get(id).getHessian(lagrange);

                if (conHs != null) {
                    hs.add((conHs));
                }
            }
            //zwiekszamy aktualny mnoznik Lagrage'a
            offset += Constraint.dbConstraint.get(id).size();
        }
        return hs;
    }

    public int getConstraintId() {
        return constraintId;
    }

    public GeometricConstraintType getConstraintType() {
        return constraintType;
    }

    @Override
    public boolean isPersistent() {
        return persistent;
    }

}
