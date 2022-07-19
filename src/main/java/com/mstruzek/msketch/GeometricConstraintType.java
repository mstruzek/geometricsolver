package com.mstruzek.msketch;

/**
 * Typ wyliczeniowy - rozne rodzaje wiezow geometrycznych
 */
public enum GeometricConstraintType {

    /**
     * Zamocowanie punktu w danym miejscu (vectorze)
     */
    FixPoint {
        @Override
        public int size() {
            return 2;
        }

        @Override
        public String getHelp() {

            return """
                Funkcja powoduje zamocowanie danego punktu w obecnym miejscu
                K - punkt do zamocowania
                """;
        }

        @Override
        public boolean isParametrized() {
            return false;
        }
    },

    /**
     * Zamocowanie punktu we wspolrzednej X określonej w parametrze.
     */
    ParametrizedXFix {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            return """
                Zamocowanie punktu w polozeniu X okreslonym wartoscia parametru P
                K - punkt ( x )
                P - sparametryzowane polozenie x
                """;
        }

        @Override
        public boolean isParametrized() {
            return true;
        }
    },

    /**
     * Zamocowanie punktu we wspolrzednej X określonej w parametrze.
     */
    ParametrizedYFix {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            return """
                Zamocowanie punktu w polozeniu Y okreslonym wartoscia parametru P
                K - punkt ( y )
                P - sparametryzowane polozenie y
                """;
        }

        @Override
        public boolean isParametrized() {
            return true;
        }
    },

    /**
     * 2 punkty maja to same polozenie - Coincidence
     */
    Connect2Points {
        @Override
        public int size() {
            return 2;
        }

        @Override
        public String getHelp() {
            return """
                Zamocowanie punktów na te same współrzędne - Coincident Constraint
                K - pierwszy punkt
                L - drugi punkt
                """;
        }

        @Override
        public boolean isParametrized() {
            return false;
        }
    },

    /**
     * 2 punkty maja to same polozenie zero-X - Coincidence
     */
    HorizontalPoint {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            return """
                Zamocowanie punktów na te same współrzędne w osi X - Coincident Constraint
                K - pierwszy punkt
                L - drugi punkt
                """;
        }

        @Override
        public boolean isParametrized() {
            return false;
        }
    },

    /**
     * 2 punkty maja to same polozenie zero-Y - Coincident
     */
    VerticalPoint {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            return """
                Zamocowanie punktów na te same współrzędne w osi Y - Coincident Constraint
                K - pierwszy punkt
                L - drugi punkt
                """;
        }

        @Override
        public boolean isParametrized() {
            return false;
        }
    },

    /**
     * Dwie linie Rownolegle - iloczyn wektorowy ,CROSS
     */
    LinesParallelism {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            return """
                Więz odpowiedzialny za równoległość dwóch lini - Parallel Constraint
                K - punkt 1 linii 1
                L - punkt 2 linii 1
                M - punkt 1 linii 2
                N - punkt 2 linii 2
                """;
        }

        @Override
        public boolean isParametrized() {
            return false;
        }
    },

    /**
     * Dwie linie Prostopadle - iloczyn skalarny , DOT
     */
    LinesPerpendicular {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            return """
                Więz odpowiedzialny za prostopadłość dwóch linii - Perpendicular Constraint
                K - punkt 1 linii 1
                L - punkt 2 linii 1
                M - punkt 1 linii 2
                N - punkt 2 linii 2
                punkty M,N -moga byc punktami nalezacymi do FixLine""";
        }

        @Override
        public boolean isParametrized() {
            return false;
        }
    },

    /**
     * Te same dlugo�ci linie
     */
    EqualLength {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            return """
                Więz zgodnej długośći dwóch lini\s
                K - punkt 1 linii 1
                L - punkt 2 linii 1
                M - punkt 1 linii 2
                N - punkt 2 linii 2
                """;
        }

        @Override
        public boolean isParametrized() {
            return false;
        }
    },

    /**
     * Wiez opisuje zaleznosc proporcji pomiedzy dlugosciami dwoch wektorów.
     */
    ParametrizedLength {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            return """
                Więz opisuje zależność relacji pomiędzy długościami dwóch wektorów opisany parametrem P.              
                K - punkt 1 linii 1
                L - punkt 2 linii 1
                M - punkt 1 linii 2
                N - punkt 2 linii 2
                P - parametr relacji względnej
                """;
        }

        @Override
        public boolean isParametrized() {
            return true;
        }
    },

    /**
     * Stycznosc okregu do prostej.
     */
    Tangency {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            return """
                Więz opisuje styczność lini do okręgu.
                K - punkt 1 na linii 1
                L - punkt 2 na linii 1
                M - srodek okregu 1
                N - promien okregu 1
                """;
        }

        @Override
        public boolean isParametrized() {
            return false;
        }

    },
    /** Stycznosc dwóch okręgów */
    CircleTangency {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            return """
                Więz opisuje stycznosc okręgu do okręgu. Także skumulowaną długość odcinków.\s
                K - srodek okregu 1
                L - srodek okregu 1
                M - srodek okregu 2
                N - promien okregu 2
                """;
        }

        @Override
        public boolean isParametrized() {
            return false;
        }

    },

    /* TERAZ WIEZY z PARAMETREM */

    /**
     * Odleglosc pomiedzy punktami
     */
    Distance2Points {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            return """
                Więz odległości określony dla punktu L od punkt K opisany parametrem P.
                K - punkt 1
                L - punkt 2
                P - parametr - dystans.
                """;
        }

        @Override
        public boolean isParametrized() {
            return true;
        }
    },

    /**
     * Odleg�o�c punktu od prostej - rownania odpowiednio jak tangency !
     */
    DistancePointLine {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            return """
                Więz odległośći punktu M od lini wyrażonej punktem K i punktem L. Dystans opisany parametrem P.
                K - punkt 1 linii 1
                L - punkt 2 linii 1
                M - punkt odlegly o P od prostej
                P - parametr
                """;
        }

        @Override
        public boolean isParametrized() {
            return true;
        }
    },

    /**
     * K�t pomiedzy dwiema prostymi
     */
    Angle2Lines {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            return """
                Więz kątowy pomiedzy wektorami opisany parametrem P.
                K - punkt 1 linii 1
                L - punkt 2 linii 1
                M - punkt 1 linii 2
                N - punkt 2 linii 2
                P - parametr ( deg )
                """;
        }

        @Override
        public boolean isParametrized() {
            return true;
        }
    },

    /**
     * Ustawia prosta horyzontalnie - rownolegle do osi X
     */
    SetHorizontal {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            return """
                Więz opisuje równoległość lini do osi współrzędnych 0-X.
                K - punkt 1 linii 1
                L - punkt 2 linii 1
                """;

        }

        @Override
        public boolean isParametrized() {
            return false;
        }
    },

    /**
     * Ustawia prosta vertycalnie - rownolegle do osi Y
     */
    SetVertical {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            return """
                Więz opisuje równoległość lini do osi 0-Y.
                K - punkt 1 linii 1
                L - punkt 2 linii 1
                """;
        }

        @Override
        public boolean isParametrized() {
            return false;
        }
    };

    public abstract int size();

    public abstract String getHelp();

    public abstract boolean isParametrized();
}
