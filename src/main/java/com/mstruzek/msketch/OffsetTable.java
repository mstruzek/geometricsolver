package com.mstruzek.msketch;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Absolut point offset from matrix 0,0 coordinates for database point set .
 */
class OffsetTable {


    private static final OffsetTable INSTANCE = new OffsetTable();

    public static OffsetTable getInstance() {
        return INSTANCE;
    }

    static int[] table;

    private OffsetTable() {
    }

    public static void setup() {
        table = new int[1 + dbPoint.keySet().stream().max(Integer::compare).orElse(0)];
        int j = 0;
        for (Integer pointId : dbPoint.keySet()) {
            table[pointId] = j;
            j++;
        }
    }

    public int pointOffset(int point) {
        return table[point];
    }
}
