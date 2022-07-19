package com.mstruzek.msketch;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Absolut point offset from matrix 0,0 coordinates for database point set .
 */
class PointLocation {

    private static final PointLocation INSTANCE = new PointLocation();

    public static PointLocation getInstance() {
        return INSTANCE;
    }

    static int[] table;

    private PointLocation() {
    }

    public static void setup() {
        table = new int[1 + dbPoint.keySet().stream().max(Integer::compare).orElse(0)];
        int j = 0;
        for (Integer pointId : dbPoint.keySet()) {
            table[pointId] = j;
            j++;
        }
    }

    /**
     * Get point location in vector space.
     * @param point id
     * @return offset location
     */
    public int get(int point) {
        return table[point];
    }
}
