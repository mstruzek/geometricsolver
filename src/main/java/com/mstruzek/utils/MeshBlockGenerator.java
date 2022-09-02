package com.mstruzek.utils;

import java.util.Locale;

import static java.lang.System.out;

public class MeshBlockGenerator {

    /**
     * @param args
     */
    public static void main(String[] args) {

        final long N = 32;
        final long P = 16;

        final double WIDTH = 50.0;
        final double HEIGHT = 50.0;

        /*
         * Guide Points numerujemy od zera
         * ========================================================================
         */
        long CONS = 0;

        for(long J = 0; J < P; J++) {


            for(long I = 0; I < N; I++) {

                /*
                 * first and second link
                 */
                long W = N * 2 * J + 2 * I + 1;
                long Z = N * 2 * J + 2 * I + 2;

                long W_p1 = getP1(W);
                long W_p2 = getP2(W);

                long Z_p1 = getP1(Z);
                long Z_p2 = getP2(Z);

                /// W
                out.printf(Locale.ROOT, "Descriptor(Point) ID(%d)    PX(%e) PY(%e);\n", W_p1, (I) * WIDTH, (J - 1) * HEIGHT);
                out.printf(Locale.ROOT, "Descriptor(Point) ID(%d)    PX(%e) PY(%e);\n", W_p2, (I) * WIDTH, (J) * HEIGHT);
                /// Z
                out.printf(Locale.ROOT, "Descriptor(Point) ID(%d)    PX(%e) PY(%e);\n", Z_p1, (I - 1) * WIDTH, (J) * HEIGHT);
                out.printf(Locale.ROOT, "Descriptor(Point) ID(%d)    PX(%e) PY(%e);\n", Z_p2, (I) * WIDTH, (J) * HEIGHT);
                //
                out.printf("\n");
                out.printf("Descriptor(GeometricObject) ID(%d)    TYPE(Line)     P1(%d)    P2(%d)    P3(-1);\n", W, W_p1, W_p2);
                CONS +=2;
                out.printf("Descriptor(GeometricObject) ID(%d)    TYPE(Line)     P1(%d)    P2(%d)    P3(-1);\n", Z, Z_p1, Z_p2);
                CONS +=2;

                /// JOIN
                out.printf("Descriptor(Constraint) ID(%d)   TYPE(Connect2Points)     K(%d)     L(%d)     M(-1)    N(-1)    PARAM(-1);\n", CONS, Z_p2,W_p2 );
                CONS +=1;
                /*
                 * =====================================================
                 */
                if(J != 0) {
                    // szukamy dolnej lapki

                    long LW = W - N * 2;
                    long LW_p2 = getP2(LW);
                    out.printf("Descriptor(Constraint) ID(%d)   TYPE(Connect2Points)     K(%d)     L(%d)     M(-1)    N(-1)    PARAM(-1);\n", CONS, W_p1,LW_p2 );
                    CONS +=1;
                }
                /*
                 * =====================================================
                 */
                if(I != 0) {
                    // szukamy bocznej lapki
                    long LZ = Z - 2; // (P2)
                    long LZ_p2 = getP2(LZ);
                    out.printf("Descriptor(Constraint) ID(%d)   TYPE(Connect2Points)     K(%d)     L(%d)     M(-1)    N(-1)    PARAM(-1);\n", CONS, Z_p1,LZ_p2 );
                    CONS +=1;
                }
            }
        }


    }

    /**
     * ========================================================================
     */
    static final int POINTS_IN_LINE = 4;

    public static long getP2(long Id) {
        return (POINTS_IN_LINE * (Id - 1)) + 2;
    }

    public static long getP1(long Id) {
        return (POINTS_IN_LINE * (Id - 1)) + 1;
    }
}
