package com.mstruzek.msketch;

import static com.mstruzek.msketch.ModelRegistry.dbPoint;

/**
 * Absolut point offset from matrix 0,0 coordinates for database point set .
 *
 * For performance reason. Even in case of large hols this table should provide performance gain.
 */
public class PointLocation {

    private static final PointLocation INSTANCE = new PointLocation();

    public static PointLocation getInstance() {
        return INSTANCE;
    }

    private PointLocation() {
    }

    private static int[] table;

    /**
     * Setup just after model is freezed before primary evaluation of Jacobian and Hessian.
     */
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
