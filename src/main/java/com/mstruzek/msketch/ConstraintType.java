package com.mstruzek.msketch;

/**
 * Typ wyliczeniowy - rozne rodzaje wiezow geometrycznych
 */
public enum ConstraintType {

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

            String description = "Zamocowanie punktu w stałym aktualnym położeniu.\n" +
                "K - punkt do zamocowania\n";
            return description;
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
            String description = "Zamocowanie punktu w polozeniu X okreslonym wartoscia parametru P\n" +
                "K - punkt ( x ) \n" +
                "P - sparametryzowane polozenie x\n";
            return description;
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
            String description = "Zamocowanie punktu w polozeniu Y okreslonym wartoscia parametru P\n" +
                "K - punkt ( y )\n" +
                "P - sparametryzowane polozenie y\n";
            return description;
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
            String description = "Zamocowanie punktów na te same współrzędne - Coincident Constraint\n" +
                "K - pierwszy punkt\n" +
                "L - drugi punkt\n";
            return description;
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
            String description = "Zamocowanie punktów na te same współrzędne w osi X - Coincident Constraint\n" +
                "K - pierwszy punkt\n" +
                "L - drugi punkt\n";
            return description;
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
            String description = "Zamocowanie punktów na te same współrzędne w osi Y - Coincident Constraint\n" +
                "K - pierwszy punkt\n" +
                "L - drugi punkt\n";
            return description;
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
            String description = "Więz odpowiedzialny za równoległość dwóch lini - Parallel Constraint\n" +
                "K - punkt 1 linii 1\n" +
                "L - punkt 2 linii 1\n" +
                "M - punkt 1 linii 2\n" +
                "N - punkt 2 linii 2\n";
            return description;
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
            String description = "Więz odpowiedzialny za prostopadłość dwóch linii - Perpendicular Constraint\n" +
                "K - punkt 1 linii 1\n" +
                "L - punkt 2 linii 1\n" +
                "M - punkt 1 linii 2\n" +
                "N - punkt 2 linii 2\n" +
                "punkty M,N -moga byc punktami nalezacymi do FixLine";
            return description;
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
            String description = "Więz zgodnej długośći dwóch lini\n" +
                "K - punkt 1 linii 1\n" +
                "L - punkt 2 linii 1\n" +
                "M - punkt 1 linii 2\n" +
                "N - punkt 2 linii 2\n";
            return description;
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
            String description = "Więz opisuje zależność relacji pomiędzy długościami dwóch wektorów opisany parametrem P.\n" +
                "K - punkt 1 linii 1\n" +
                "L - punkt 2 linii 1\n" +
                "M - punkt 1 linii 2\n" +
                "N - punkt 2 linii 2\n" +
                "P - parametr relacji względnej\n";
            return description;
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
            String description = "Więz opisuje styczność lini do okręgu.\n" +
                "K - punkt 1 na linii 1\n" +
                "L - punkt 2 na linii 1\n" +
                "M - srodek okregu 1\n" +
                "N - promien okregu 1\n";
            return description;
        }

        @Override
        public boolean isParametrized() {
            return false;
        }

    },
    /**
     * Stycznosc dwóch okręgów
     */
    CircleTangency {
        @Override
        public int size() {
            return 1;
        }

        @Override
        public String getHelp() {
            String description = "Więz opisuje stycznosc okręgu do okręgu. Także skumulowaną długość odcinków.\n" +
                "K - srodek okregu 1\n" +
                "L - srodek okregu 1\n" +
                "M - srodek okregu 2\n" +
                "N - promien okregu 2\n";
            return description;
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
            String description = "Więz odległości określony dla punktu L od punkt K opisany parametrem P.\n" +
                "K - punkt 1\n" +
                "L - punkt 2\n" +
                "P - parametr - dystans.\n";
            return description;
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
            String description = "Więz odległośći punktu M od lini wyrażonej punktem K i punktem L. Dystans opisany parametrem P.\n" +
                "K - punkt 1 linii 1\n" +
                "L - punkt 2 linii 1\n" +
                "M - punkt odlegly o P od prostej\n" +
                "P - parametr\n";
            return description;
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
            String description = "Więz kątowy pomiedzy wektorami opisany parametrem P.\n" +
                "K - punkt 1 linii 1\n" +
                "L - punkt 2 linii 1\n" +
                "M - punkt 1 linii 2\n" +
                "N - punkt 2 linii 2\n" +
                "P - parametr ( deg )\n";
            return description;
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
            String description = "Więz opisuje równoległość lini do osi współrzędnych 0-X.\n" +
                "K - punkt 1 linii 1\n" +
                "L - punkt 2 linii 1\n";

            return description;
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
            String description = "Więz opisuje równoległość lini do osi 0-Y.\n" +
                "K - punkt 1 linii 1\n" +
                "L - punkt 2 linii 1\n";
            return description;
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
