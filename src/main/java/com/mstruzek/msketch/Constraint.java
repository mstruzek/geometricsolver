package com.mstruzek.msketch;

import java.util.TreeMap;

import com.mstruzek.msketch.matrix.BindMatrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

/*
 * Wiez jedowymairowy,iloczyn skalarny,
 */
public abstract class Constraint implements ConstraintInterface {


    /**
     * Licznik wiezow
     */
    static int counter = 0;

    /**
     * numer kolejno utworzonej lini
     */
    protected int constraintId;

    protected GeometricConstraintType constraintType = null;

    /**
     * tablica wszystkich linii
     */
    public static TreeMap<Integer, Constraint> dbConstraint = new TreeMap<Integer, Constraint>();


    public Constraint(Integer constraintId, GeometricConstraintType constraintType) {
        super();
        this.constraintId = constraintId;
        this.constraintType = constraintType;
        dbConstraint.put(constraintId,this);
    }

    public static Integer nextId(){
        return counter++;
    }

    /**
     * Funkcja zwraca WARTOSC WIEZU w postaci skalaru ,przypadek dla Constraint.size=1
     * lub Vectora kolumnowego gdy size=2;
     * bedziemy uzywac podczas sprawdzania czy wiez jest wyzerowany
     *
     * @param dbPoints    wszystkie ruchome punkty w naszym zadaniu
     * @param dbParameter baza wszystkich parametrow
     * @return macierz albo 1x1 albo 2x1;
     */
    public abstract MatrixDouble getValue(TreeMap<Integer, Point> dbPoints, TreeMap<Integer, Parameter> dbParameter);

    /**
     * Funkcja zwraca Jacobian w postaci wektora wierszowego gdy Constraint.size=1,
     * lub dwa wektory wierszowe gdy size=2;
     *
     * @param dbPoints    wszystkie ruchome punkty w naszym zadaniu
     * @param dbParameter baza wszystkich parametrow
     * @return Funkcja zwraca wktor wierszowy o dlugosci proporcjonalnej do ilosci Point'ow
     */
    public abstract MatrixDouble getJacobian(TreeMap<Integer, Point> dbPoints, TreeMap<Integer, Parameter> dbParameter);

    /**
     * Funkcja zwraca norme danego wiezu
     *
     * @param dbPoints
     * @param dbParameter
     * @return
     */
    public abstract double getNorm(TreeMap<Integer, Point> dbPoints, TreeMap<Integer, Parameter> dbParameter);

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
    public abstract MatrixDouble getHessian(TreeMap<Integer, Point> dbPoints, TreeMap<Integer, Parameter> dbParameter);

    /**
     * Funkcja zwraca true jesli Hessian jest staly
     *
     * @return
     */
    public abstract boolean isHessianConstant();

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
        int out = 0;
        for (Integer i : dbConstraint.keySet()) {
            out += dbConstraint.get(i).size();
        }
        return out;
    }

    /**
     * Funkcja zwraca Jakobian ze wszystkich wiezow
     *
     * @param dbPoints
     * @param dbParameter
     * @return macierz ,jakobian wiezow d(Wiezy)/dq
     */
    public static MatrixDouble getFullJacobian(TreeMap<Integer, Point> dbPoints, TreeMap<Integer, Parameter> dbParameter) {
        MatrixDouble jak = MatrixDouble.fill(allLagrangeCoffSize(), dbPoints.size() * 2, 0.0);
        int rowPos = 0;
        for (Integer i : dbConstraint.keySet()) {
            jak.addSubMatrix(rowPos, 0, dbConstraint.get(i).getJacobian(dbPoints, dbParameter));
            rowPos += dbConstraint.get(i).size();
        }
        return jak;
    }

    /**
     * Funkcja zwraca prawe strony , czyli wartosci wszystkich wiezow
     * w wektorze kolumnowym
     *
     * @param dbPoints
     * @param dbParameter
     * @return
     */
    public static MatrixDouble getFullConstraintValues(TreeMap<Integer, Point> dbPoints, TreeMap<Integer, Parameter> dbParameter) {
        MatrixDouble wi = MatrixDouble.fill(allLagrangeCoffSize(), 1, 0.0);
        int currentRow = 0;
        for (Integer i : dbConstraint.keySet()) {
            wi.addSubMatrix(currentRow, 0, dbConstraint.get(i).getValue(dbPoints, dbParameter));
            currentRow += dbConstraint.get(i).size();
        }
        return wi;
    }

    /**
     * Funkcja zwraca calkowita norme dla wszystkich wiezow
     *
     * @param dbPoints
     * @param dbParameter
     * @return
     */
    public static double getFullNorm(TreeMap<Integer, Point> dbPoints, TreeMap<Integer, Parameter> dbParameter) {
        double norm = 0;
        double tmp = 0;
        for (Integer i : dbConstraint.keySet()) {
            tmp = dbConstraint.get(i).getNorm(dbPoints, dbParameter);
            norm += tmp * tmp;
        }

        return Math.sqrt(norm);
    }

    /**
     * Funkcja zwraca macierz d(Jak'*a)/dq -czyli tak jakby calkowity hessian juz
     * przemnozony
     *
     * @param dbPoints
     * @param dbParameter
     * @param bMX         - wektor x , z niego wyciagniemy mnozniki lagrange'a
     * @return
     */
    public static MatrixDouble getFullHessian(TreeMap<Integer, Point> dbPoints, TreeMap<Integer, Parameter> dbParameter, BindMatrix bMX) {

        // Full  HESSIAN
        MatrixDouble HSn = MatrixDouble.fill(dbPoints.size() * 2, dbPoints.size() * 2, 0.0);

        int aCounter = 0; //licznik mnoznikow lagrange'a

        double currentA = 0.0;//wartosc aktualnego mnoznika

        // Po wszystkich wiezach
        for (Integer i : dbConstraint.keySet()) {
            if (!(dbConstraint.get(i).isJacobianConstant())) {
                //jest hessian
                currentA = bMX.get(Point.dbPoint.size() * 2 + aCounter, 0);

                //
                //  Hessian - dla tego wiezu liczony na cala macierz !
                // -- ! add into mem in place AddVisitator
                //
                MatrixDouble Hs = dbConstraint.get(i).getHessian(dbPoints, dbParameter).dot(currentA);
                HSn.add((Hs));
            }
            //zwiekszamy aktualny mnoznik Lagrage'a
            aCounter += dbConstraint.get(i).size();
        }

        return HSn;
    }

    /**
     * Usun wiez o danym id
     *
     * @param id - wiezu
     */
    public static void remove(int id) {
        dbConstraint.remove(id);
    }

    public int getConstraintId() {
        return constraintId;
    }

    public GeometricConstraintType getConstraintType() {
        return constraintType;
    }



}
