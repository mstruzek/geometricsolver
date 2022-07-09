package com.mstruzek.msketch;

import java.util.Set;
import java.util.TreeMap;

import com.mstruzek.msketch.matrix.BindMatrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

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

    /** [ false ] - this constraint is normally not visible unless CTRL function applied into layout */
    protected boolean storable;

    /**
     * tablica wszystkich linii
     */
    public static TreeMap<Integer, Constraint> dbConstraint = new TreeMap<Integer, Constraint>();


    public Constraint(Integer constraintId, GeometricConstraintType constraintType, boolean storable) {
        super();
        this.constraintId = constraintId;
        this.constraintType = constraintType;
        this.storable = storable;
        dbConstraint.put(constraintId,this);
    }

    public static Integer nextId(){
        return constraintCounter++;
    }

    public static Integer nextId(Set<Integer> skipIdentifiers){
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
     * @return Funkcja zwraca wktor wierszowy o dlugosci proporcjonalnej do ilosci Point'ow
     */
    public abstract MatrixDouble getJacobian();

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
    public abstract MatrixDouble getHessian();

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
     * @return macierz ,jakobian wiezow d(Wiezy)/dq
     */
    public static MatrixDouble getFullJacobian() {
        MatrixDouble jak = MatrixDouble.fill(allLagrangeCoffSize(), dbPoint.size() * 2, 0.0);
        int rowPos = 0;
        for (Integer i : dbConstraint.keySet()) {
            jak.addSubMatrix(rowPos, 0, dbConstraint.get(i).getJacobian());
            rowPos += dbConstraint.get(i).size();
        }
        return jak;
    }

    /**
     * Funkcja zwraca prawe strony , czyli wartosci wszystkich wiezow
     * w wektorze kolumnowym
     *
     * @return
     */
    public static MatrixDouble getFullConstraintValues() {
        MatrixDouble wi = MatrixDouble.fill(allLagrangeCoffSize(), 1, 0.0);
        int currentRow = 0;
        for (Integer i : dbConstraint.keySet()) {
            wi.addSubMatrix(currentRow, 0, dbConstraint.get(i).getValue());
            currentRow += dbConstraint.get(i).size();
        }
        return wi;
    }

    /**
     * Funkcja zwraca calkowita norme dla wszystkich wiezow
     *
     * @return
     */
    public static double getFullNorm() {
        double norm = 0;
        double tmp = 0;
        for (Integer i : dbConstraint.keySet()) {
            tmp = dbConstraint.get(i).getNorm();
            norm += tmp * tmp;
        }

        return Math.sqrt(norm);
    }

    /**
     * Funkcja zwraca macierz d(Jak'*a)/dq -czyli tak jakby calkowity hessian juz
     * przemnozony
     *
     * @param bMX - wektor x , z niego wyciagniemy mnozniki lagrange'a
     * @return
     */
    public static MatrixDouble getFullHessian(BindMatrix bMX) {

        // Full  HESSIAN
        MatrixDouble HSn = MatrixDouble.fill(dbPoint.size() * 2, dbPoint.size() * 2, 0.0);

        int aCounter = 0; //licznik mnoznikow lagrange'a

        double currentA = 0.0;//wartosc aktualnego mnoznika

        // Po wszystkich wiezach
        for (Integer i : dbConstraint.keySet()) {
            if (!(dbConstraint.get(i).isJacobianConstant())) {
                //jest hessian
                currentA = bMX.get(dbPoint.size() * 2 + aCounter, 0);

                //
                //  Hessian - dla tego wiezu liczony na cala macierz !
                // -- ! add into mem in place AddVisitator
                //
                MatrixDouble Hs = dbConstraint.get(i).getHessian().dot(currentA);
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


    @Override
    public boolean isStorable(){
        return storable;
    }
}
