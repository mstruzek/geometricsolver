package com.mstruzek.utils;

import java.util.Locale;

import static java.lang.System.out;

public class BlockGeneratorTest3 {

    static final long N = 2;
    static final long P = 2;

    static final double WIDTH = 50.0;
    static final double HEIGHT = 50.0;

    static final boolean ENABLE_PARALLEL_CONSTRAINT = true;

    static final int NO_GEOM = 2;

    /**
     * Parallel Constraints
     * @param args
     */
    public static void main(String[] args) {
        /// Guide Points numerujemy od zera
        out.printf("#--signature: GeometricConstraintSolver  2009-2022\n");
        out.printf("#--file-format: V1\n");
        out.printf("\n");
        out.printf("#--definition-begin: ;\n");
        out.printf("\n");

        long CONS = 0;
        for (long J = 0; J < P; J++) {
            for (long I = 0; I < N; I++) {
                ///  first and second link
                long LW = N * NO_GEOM * J + NO_GEOM * I + 1;     // lower vertical line
                long LZ = N * NO_GEOM * J + NO_GEOM * I + 2;     // upper horizontal line

                long LW_p1 = getP1(LW);
                long LW_p2 = getP2(LW);

                long LZ_p1 = getP1(LZ);
                long LZ_p2 = getP2(LZ);

                /// LW
                out.printf(Locale.ROOT, "Descriptor(Point) ID(%d)    PX(%e) PY(%e);\n", LW_p1, (I) * WIDTH, (J - 1) * HEIGHT);
                out.printf(Locale.ROOT, "Descriptor(Point) ID(%d)    PX(%e) PY(%e);\n", LW_p2, (I) * WIDTH, (J) * HEIGHT);
                /// LZ
                out.printf(Locale.ROOT, "Descriptor(Point) ID(%d)    PX(%e) PY(%e);\n", LZ_p1, (I - 1) * WIDTH, (J) * HEIGHT);
                out.printf(Locale.ROOT, "Descriptor(Point) ID(%d)    PX(%e) PY(%e);\n", LZ_p2, (I) * WIDTH, (J) * HEIGHT);
                //
                out.printf("\n");
                out.printf("Descriptor(GeometricObject) ID(%d)    TYPE(Line)     P1(%d)    P2(%d)    P3(-1);\n", LW, LW_p1, LW_p2);
                CONS += 2;
                out.printf("Descriptor(GeometricObject) ID(%d)    TYPE(Line)     P1(%d)    P2(%d)    P3(-1);\n", LZ, LZ_p1, LZ_p2);
                CONS += 2;
                /// JOIN
                out.printf("Descriptor(Constraint) ID(%d)   TYPE(Connect2Points)     K(%d)     L(%d)     M(-1)    N(-1)    PARAM(-1);\n", CONS, LZ_p2, LW_p2);
                CONS += 1;

                /// -----------------------------------------------
                if (J != 0) {
                    // szukamy dolnej lapki
                    long pLW = LW - N * NO_GEOM;
                    long pLW_p1 = getP1(pLW);
                    long pLW_p2 = getP2(pLW);
                    out.printf("Descriptor(Constraint) ID(%d)   TYPE(Connect2Points)     K(%d)     L(%d)     M(-1)    N(-1)    PARAM(-1);\n", CONS, LW_p1, pLW_p2);
                    CONS += 1;
                }
                /// -----------------------------------------------
                if (ENABLE_PARALLEL_CONSTRAINT && J != 0) {
                    long pLW = LW - N * NO_GEOM;
                    long pLW_p1 = getP1(pLW);
                    long pLW_p2 = getP2(pLW);
                    out.printf("Descriptor(Constraint) ID(%d)   TYPE(LinesParallelism)     K(%d)     L(%d)     M(%d)    N(%d)    PARAM(-1);\n", CONS, LW_p1, LW_p2, pLW_p1, pLW_p2);
                    CONS += 1;
                }
                /// -----------------------------------------------
                if (I != 0) {
                    // szukamy bocznej lapki
                    long pLZ = LZ - NO_GEOM; // (P2)
                    long pLZ_p2 = getP2(pLZ);
                    out.printf("Descriptor(Constraint) ID(%d)   TYPE(Connect2Points)     K(%d)     L(%d)     M(-1)    N(-1)    PARAM(-1);\n", CONS, LZ_p1, pLZ_p2);
                    CONS += 1;
                }
                /// -----------------------------------------------
                if (ENABLE_PARALLEL_CONSTRAINT && I != 0) {
                    long pLZ = LZ - NO_GEOM; // (P2)
                    long pLZ_p1 = getP1(pLZ);
                    long pLZ_p2 = getP2(pLZ);
                    out.printf("Descriptor(Constraint) ID(%d)   TYPE(LinesParallelism)     K(%d)     L(%d)     M(%d)    N(%d)    PARAM(-1);\n", CONS, LZ_p1, LZ_p2, pLZ_p1, pLZ_p2);
                    CONS += 1;
                }
                /// -----------------------------------------------
            }
        }
        out.printf("\n");
        out.printf("#--definition-end: ;\n");
        out.printf("\n");
    }

    /**
     * ========================================================================
     */
    static final int POINTS_IN_LINE = 2 * 2;

    public static long getP2(long Id) {
        return (POINTS_IN_LINE * (Id - 1)) + 2;
    }

    public static long getP1(long Id) {
        return (POINTS_IN_LINE * (Id - 1)) + 1;
    }
}
