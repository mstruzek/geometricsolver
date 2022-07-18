package com.mstruzek.msketch;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Absolut point offset from matrix 0,0 coordinates for database point set .
 */
class VectorSpace {

    private static final VectorSpace INSTANCE = new VectorSpace();

    public static VectorSpace getInstance() {
        return INSTANCE;
    }

    static int[] table;

    private VectorSpace() {
    }

    public static void setup() {
        table = new int[1 + dbPoint.keySet().stream().max(Integer::compare).orElse(0)];
        int j = 0;
        for (Integer pointId : dbPoint.keySet()) {
            table[pointId] = j;
            j++;
        }
    }

    public int pointIndex(int point) {
        return table[point];
    }
}
