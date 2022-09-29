package com.mstruzek.graphic;

import java.util.*;

/**
 * Structure for fast search with approximation in double view space . Expected behaviour as in RTree map.
 * Each coordinate is stored in its own treemap structure and treated as an index.
 * Append only interface provided. No removal supported.
 * Performance characteristic:
 * 10_000 elements - 15 - 16ms
 * 100 search rounds  < max 16 ms
 */
public class GeometricSOAIndex {

    public static final int SLOT_BOUNDING = 10;
    private final TreeMap<Double, Integer> mapX = new TreeMap<>();
    private final TreeMap<Double, Integer> mapY = new TreeMap<>();

    private final double nextSlot = 1.0e-9; // 10.0e6 max double value

    private static final int SEARCH_ROUND_LIMIT = SLOT_BOUNDING;

    public GeometricSOAIndex() {
    }

    /**
     * Add geometric point at coordinates into SoA index !
     * @param id point id
     * @param x  coordinate x
     * @param y  coordinate y
     */
    public void addGeometricPoint(Integer id, double x, double y) {
        addGeometricPointYCoordinate(y, id);
        addGeometricPointXCoordinate(x, id);
    }

    /**
     * Find with minimal approximation all points around point cx, cy with in bounding box.
     * Bounding box boundaries evaluated from distance.
     * @param cx       bounding box X coordinate
     * @param cy       bounding box Y coordinate
     * @param distance width and height
     * @return point set from this bounding box
     */
    public Set<Integer> findWithInBoundingBox(double cx, double cy, double distance) {
        Set<Integer> pointsOnXAxis = findWithinXAxis(cx, distance);
        Set<Integer> pointsOnYAxis = findWithinYAxis(cy, distance);
        pointsOnXAxis.retainAll(pointsOnYAxis);
        return pointsOnXAxis;
    }

    private void addGeometricPointYCoordinate(double y, int id) {
        double approx = y;
        int maxSearchRound = SEARCH_ROUND_LIMIT;
        while (mapY.get(approx) != null && (maxSearchRound--) > 0) {
            approx = approx + nextSlot;
        }
        if (maxSearchRound == 0) throw new IllegalStateException("above search round limit 10");
        mapY.put(approx, id);
    }

    private void addGeometricPointXCoordinate(double x, int id) {
        double approx = x;
        int searchRound = SEARCH_ROUND_LIMIT;
        while (mapX.get(approx) != null && (searchRound--) > 0) {
            approx = approx + nextSlot;
        }
        if (searchRound == 0) throw new IllegalStateException("above search round limit 10");
        mapX.put(approx, id);
    }

    private Set<Integer> findWithinXAxis(double cx, double distance) {
        SortedMap<Double, Integer> navigableMap = mapX.subMap(cx - distance, cx + distance + nextSlot);
        return new HashSet<>(navigableMap.values());
    }

    private Set<Integer> findWithinYAxis(double cy, double distance) {
        SortedMap<Double, Integer> navigableMap = mapY.subMap(cy - distance, cy + distance + nextSlot);
        return new HashSet<>(navigableMap.values());
    }
}
