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
    public abstract MatrixDouble getHessian(double alfa);

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
     * @param mt
     */
    public static void getFullJacobian(MatrixDouble mt) {
        int rowPos = 0;
        for (Integer i : dbConstraint.keySet()) {
            mt.setSubMatrix(rowPos, 0, dbConstraint.get(i).getJacobian());
            rowPos += dbConstraint.get(i).size();
        }
    }

    /**
     * Funkcja zwraca prawe strony , czyli wartosci wszystkich wiezow
     * w wektorze kolumnowym
     *
     * @return
     * @param mt
     */
    public static void getFullConstraintValues(MatrixDouble mt) {
        int currentRow = 0;
        for (Integer i : dbConstraint.keySet()) {
            mt.setSubMatrix(currentRow, 0, dbConstraint.get(i).getValue());
            currentRow += dbConstraint.get(i).size();
        }
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
     * @param hs - Full  HESSIAN
     * @param dmx  - wektor x , z niego wyciagniemy mnozniki lagrange'a
     * @return
     */
    public static MatrixDouble getFullHessian(MatrixDouble hs, BindMatrix dmx) {

        int offset = 0; //licznik mnoznikow lagrange'a
        double alfa = 0.0;//wartosc aktualnego mnoznika

        // Po wszystkich wiezach
        for (Integer i : dbConstraint.keySet()) {
            if (!(dbConstraint.get(i).isJacobianConstant())) {
                //jest hessian
                alfa = dmx.get(dbPoint.size() * 2 + offset, 0);

                //
                //  Hessian - dla tego wiezu liczony na cala macierz !
                // -- ! add into mem in place AddVisitator
                //
                MatrixDouble Hs = dbConstraint.get(i).getHessian(alfa); /// FIXME -- alfa not implemented
                if (Hs != null) {
                    hs.add((Hs));
                }
            }
            //zwiekszamy aktualny mnoznik Lagrage'a
            offset += dbConstraint.get(i).size();
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
    public boolean isStorable(){
        return storable;
    }
}
