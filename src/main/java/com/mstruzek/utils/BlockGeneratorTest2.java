package com.mstruzek.utils;

import java.util.Locale;

import static java.lang.System.out;

public class BlockGeneratorTest2 {

    static final long N = 5;
    static final long P = 3;

    static final double WIDTH = 50.0;
    static final double HEIGHT = 50.0;

    static final boolean ENABLE_PERPENDICULAR_CONSTRAINT = false;
    static final boolean ENABLE_TANGENCY_CONSTRAINT = true;

    static final double CIRCLE_RADIUS = WIDTH * 0.4 / 2;

    static final int NO_GEOM = 3;
    //static final int NO_GEOM = 3;

    static final Locale lc = Locale.ROOT;

    /**
     * Perpendicaul Constraint with Tangent Circles !
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
                long W = N * NO_GEOM * J + NO_GEOM * I + 1;     // lower vertical line
                long Z = N * NO_GEOM * J + NO_GEOM * I + 2;     // upper horizontal line
                long C = N * NO_GEOM * J + NO_GEOM * I + 3;     // circle

                long LW_p1 = getP1(W);
                long LW_p2 = getP2(W);

                long LZ_p1 = getP1(Z);
                long LZ_p2 = getP2(Z);

                long LC_p1 = getP1(C);
                long LC_p2 = getP2(C);

                out.printf(lc, "Descriptor(Point) ID(%d)    PX(%e) PY(%e);\n", LW_p1, (I) * WIDTH, (J - 1) * HEIGHT);
                out.printf(lc, "Descriptor(Point) ID(%d)    PX(%e) PY(%e);\n", LW_p2, (I) * WIDTH, (J) * HEIGHT);
                /// Z
                out.printf(lc, "Descriptor(Point) ID(%d)    PX(%e) PY(%e);\n", LZ_p1, (I - 1) * WIDTH, (J) * HEIGHT);
                out.printf(lc, "Descriptor(Point) ID(%d)    PX(%e) PY(%e);\n", LZ_p2, (I) * WIDTH, (J) * HEIGHT);

                // circle center P1
                out.printf(lc, "Descriptor(Point) ID(%d)    PX(%e) PY(%e);\n", LC_p1,
                    (I - 1) * WIDTH  + WIDTH/ 2,
                    (J - 1) * HEIGHT  + HEIGHT/2);
                // radius P2
                out.printf(lc, "Descriptor(Point) ID(%d)    PX(%e) PY(%e);\n", LC_p2,
                    (I - 1) * WIDTH  + WIDTH/2 + CIRCLE_RADIUS,
                    (J - 1) * HEIGHT  + HEIGHT/2 + CIRCLE_RADIUS);

                //
                out.printf("\n");
                out.printf("Descriptor(GeometricObject) ID(%d)    TYPE(Line)     P1(%d)    P2(%d)    P3(-1);\n", W, LW_p1, LW_p2);
                CONS += 2;
                out.printf("Descriptor(GeometricObject) ID(%d)    TYPE(Line)     P1(%d)    P2(%d)    P3(-1);\n", Z, LZ_p1, LZ_p2);
                CONS += 2;
                out.printf("Descriptor(GeometricObject) ID(%d)    TYPE(Circle)     P1(%d)    P2(%d)    P3(-1);\n", C, LC_p1, LC_p2);
                CONS += 2;

                /// JOIN
                out.printf("Descriptor(Constraint) ID(%d)   TYPE(Connect2Points)     K(%d)     L(%d)     M(-1)    N(-1)    PARAM(-1);\n", CONS, LZ_p2, LW_p2);
                CONS += 1;

                /// -----------------------------------------------
                if (J != 0) {
                    // szukamy dolnej lapki
                    long pLW = W - N * NO_GEOM;
                    long pLW_p2 = getP2(pLW);
                    out.printf("Descriptor(Constraint) ID(%d)   TYPE(Connect2Points)     K(%d)     L(%d)     M(-1)    N(-1)    PARAM(-1);\n", CONS, LW_p1, pLW_p2);
                    CONS += 1;
                }
                /// -----------------------------------------------
                if (I != 0) {
                    // szukamy bocznej lapki
                    long pLZ = Z - NO_GEOM; // (P2)
                    long pLZ_p2 = getP2(pLZ);
                    out.printf("Descriptor(Constraint) ID(%d)   TYPE(Connect2Points)     K(%d)     L(%d)     M(-1)    N(-1)    PARAM(-1);\n", CONS, LZ_p1, pLZ_p2);
                    CONS += 1;
                }
                /// -----------------------------------------------
                if (ENABLE_PERPENDICULAR_CONSTRAINT) {
                    out.printf("Descriptor(Constraint) ID(%d)   TYPE(LinesPerpendicular)     K(%d)     L(%d)     M(%d)    N(%d)    PARAM(-1);\n", CONS, LW_p1, LW_p2, LZ_p1, LZ_p2);
                    CONS += 1;
                }
                /// -----------------------------------------------
                if (ENABLE_TANGENCY_CONSTRAINT) {
                    /// tangency
                    out.printf("Descriptor(Constraint) ID(%d)   TYPE(Tangency)     K(%d)     L(%d)     M(%d)    N(%d)    PARAM(-1);\n", CONS, LW_p1, LW_p2, LC_p1, LC_p2);
                    CONS += 1;
                    out.printf("Descriptor(Constraint) ID(%d)   TYPE(Tangency)     K(%d)     L(%d)     M(%d)    N(%d)    PARAM(-1);\n", CONS, LZ_p1, LZ_p2, LC_p1, LC_p2);
                    CONS += 1;
                }
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
