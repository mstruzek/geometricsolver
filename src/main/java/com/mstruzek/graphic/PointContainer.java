package com.mstruzek.graphic;

/**
 * Klasa przechowujaca numery punktow ktore chcemy
 * wykorzystac poczas tworzenia wiezow
 *
 * @author root
 */
public class PointContainer {

    int PointK = -1;
    int PointL = -1;
    int PointM = -1;
    int PointN = -1;

    public PointContainer(int pointK, int pointL, int pointM, int pointN) {
        super();
        PointK = pointK;
        PointL = pointL;
        PointM = pointM;
        PointN = pointN;
    }

    public int getPointK() {
        return PointK;
    }

    public void setPointK(int pointK) {
        PointK = pointK;
    }

    public int getPointL() {
        return PointL;
    }

    public void setPointL(int pointL) {
        PointL = pointL;
    }

    public int getPointM() {
        return PointM;
    }

    public void setPointM(int pointM) {
        PointM = pointM;
    }

    public int getPointN() {
        return PointN;
    }

    public void setPointN(int pointN) {
        PointN = pointN;
    }

    /**
     * Czysci konkretny punkt K
     */
    public void clearK() {
        PointK = -1;
    }

    /**
     * Czysci konkretny punkt L
     */
    public void clearL() {
        PointL = -1;
    }

    /**
     * Czysci konkretny punkt M
     */
    public void clearM() {
        PointM = -1;
    }

    /**
     * Czysci konkretny punkt N
     */
    public void clearN() {
        PointN = -1;
    }

    public void setFreeSlot(int pointId) {
        if (PointK == -1) {
            PointK = pointId;
        } else if (PointL == -1) {
            PointL = pointId;
        } else if (PointM == -1) {
            PointM = pointId;
        } else if (PointN == -1) {
            PointN = pointId;
        } else {
            clearAll();
            PointK = pointId;
        }
    }

    /**
     * Czysci wszystkie punkty
     */
    public void clearAll() {
        PointK = -1;
        PointL = -1;
        PointM = -1;
        PointN = -1;
    }

    @Override
    public String toString() {
        String out = "";
        if (PointK >= 0) {
            out += " K - " + PointK + "  ";
        }
        if (PointL >= 0) {
            out += " L - " + PointL + "  ";
        }
        if (PointM >= 0) {
            out += " M - " + PointM + "  ";
        }
        if (PointN >= 0) {
            out += " N - " + PointN + "  ";
        }
        return out;
    }
}
