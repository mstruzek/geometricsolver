package com.mstruzek.sparsematrixsolver;

import java.util.SortedMap;
import java.util.TreeMap;

/**
 * Klasa reprezentuje pozycje danego elementu w macierzy rzadkiej
 *
 * @author root
 */
public class Index implements Comparable<Index> {
    private int x;
    private int y;

    public Index(int x, int y) {
        super();
        this.x = x;
        this.y = y;
    }

    public int compareTo(Index index) {
        int ix = index.x;

        if (ix == x) {
            int iy = index.y;
            if (iy == y) {
                return 0;
            } else if (iy < y) {
                return -1;
            } else {
                return 1;
            }
        } else if (ix < x) {
            return -1;
        } else {
            return 1;
        }
    }

    public int hashCode() {
        final int PRIME = 31;
        int result = 1;
        result = PRIME * result + (int) (x ^ (x >>> 32));
        result = PRIME * result + (int) (y ^ (y >>> 32));
        return result;
    }

    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        final Index other = (Index) obj;
        if (x != other.x)
            return false;
        if (y != other.y)
            return false;
        return true;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    /**
     * Zamienia x z y miejscami
     *
     * @return
     */
    public Index transposeC() {
        return new Index(y, x);
    }

    public String toString() {
        return x + "," + y;
    }

    public static void main(String[] args) {

        SortedMap<Index, Double> entries = new TreeMap<Index, Double>();
        entries.put(new Index(1, 4), 1e-4);
        entries.put(new Index(2, 1000), 5555.0);
        entries.put(new Index(1, 1000), 555.0);
        entries.put(new Index(1, 10), 5555.0);


        for (Index id : entries.keySet())
            System.out.println(id);

        System.out.println(entries);

    }
}

