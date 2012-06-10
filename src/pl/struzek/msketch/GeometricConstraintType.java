package pl.struzek.msketch;

/**
 * Typ wyliczeniowy - rozne rodzaje wiezow geometrycznych
 *
 */
public enum GeometricConstraintType {

	/** Zamocowanie punktu w danym miejscu (vectorze) */
	FixPoint {
		@Override
		public
		int size() {
			return 2;
		}

		@Override
		public
		String getHelp() {
			
			return "Funkcja powoduje zamocowanie danego punktu w obecnym miejscu\n" +
					"K - punkt do zamocowania";
		}
	},
	/** 2 punkty maja to same polozenie - Coincidence*/
	Conect2Points {
		@Override
		public
		int size() {
			return 2;
		}

		@Override
		public
		String getHelp() {
			return "Dwa punkty maja to samo polozenie , czyli tak zwany wiez Coincidence\n" +
					"K - pierwszy punkt\n" +
					"L - drugi punkt\n";
		}
	},
	/** Dwie linie Rownolegle - iloczyn wektorowy ,CROSS */
	LinesParallelism {
		@Override
		public
		int size() {
			return 1;
		}

		@Override
		public
		String getHelp() {
			return "Wiez odpowiedzialny za rownoleglosc dwoch linii\n" +
					"K - punkt 1 linii 1\n" +
					"L - punkt 2 linii 1\n" +
					"M - punkt 1 linii 2\n" +
					"N - punkt 2 linii 2\n" +
					"punkty M,N -moga byc punktami nalezacymi do FixLine";
		}
	},
	/** Dwie linie Prostopadle - iloczyn skalarny , DOT*/
	LinesPerpendicular {
		@Override
		public
		int size() {
			return 1;
		}

		@Override
		public
		String getHelp() {
			return "Wiez odpowiedzialny za prostopadlosc dwoch linii\n" +
			"K - punkt 1 linii 1\n" +
			"L - punkt 2 linii 1\n" +
			"M - punkt 1 linii 2\n" +
			"N - punkt 2 linii 2\n" +
			"punkty M,N -moga byc punktami nalezacymi do FixLine";
		}
	},
	/** Te same dlugoœci linie  */
	LinesSameLength {
		@Override
		public
		int size() {
			return 1;
		}

		@Override
		public
		String getHelp() {
			return "Dlugosc 1 linii = Dlugosci 2 linii \n" +
			"K - punkt 1 linii 1\n" +
			"L - punkt 2 linii 1\n" +
			"M - punkt 1 linii 2\n" +
			"N - punkt 2 linii 2\n";
		}
	},
	/** Stycznosc okregu do prostej*/
	Tangency{
		@Override
		public
		int size() {
			return 1;
		}

		@Override
		public
		String getHelp() {
			return "Stycznosc okregu do linii \n" +
			"K - punkt 1 linii 1\n" +
			"L - punkt 2 linii 1\n" +
			"M - srodek okregu \n" +
			"N - promien okregu\n";
		}
		
	},
	
	/** TERAZ WIEZY z PARAMETREM */
	
	/** Odleglosc pomiedzy punktami */
	Distance2Points {
		@Override
		public
		int size() {
			return 1;
		}

		@Override
		public
		String getHelp() {
			return "Odleglosc 2 punktow sparametryzowana \n" +
			"K - punkt 1 \n" +
			"L - punkt 2\n" +
			"P - parametr\n";
		}
	},
	/** Odleg³oœc punktu od prostej FIXME - zrobic ten ponizej */
	DistancePointLine {
		@Override
		public
		int size() {
			return 1;
		}

		@Override
		public
		String getHelp() {
			return "Odleglosc punktu od prostej sparametryzowana \n" +
			"K - punkt 1 linii 1 \n" +
			"L - punkt 2 linii 1\n" +
			"M - punkt odlegly od prostej o parametr P\n" +
			"P - parametr\n";
		}
	},
	/** K¹t pomiedzy dwiema prostymi*/
	Angle2Lines {
		@Override
		public
		int size() {
			return 1;
		}

		@Override
		public
		String getHelp() {
			return "Kat pomiedzy dwiema prostymi \n" +
			"K - punkt 1 linii 1\n" +
			"L - punkt 2 linii 1\n" +
			"M - punkt 1 linii 2\n" +
			"N - punkt 2 linii 2\n" +
			"P - parametr ,wartosc kata\n";
		}
	},
	/** Ustawia prosta horyzontalnie - rownolegle do osi X*/
	SetHorizontal{
		@Override
		public
		int size() {
			return 1;
		}

		@Override
		public
		String getHelp() {
			return "Ustawia linie rownolegle do osi X \n" +
			"K - punkt 1 linii 1\n" +
			"L - punkt 2 linii 1\n" ;

		}		
	},
	/**Ustawia prosta vertycalnie - rownolegle do osi Y */
	SetVertical{
		@Override
		public
		int size() {
			return 1;
		}

		@Override
		public
		String getHelp() {
			return "Ustawia linie rownolegle do osi Y \n" +
			"K - punkt 1 linii 1\n" +
			"L - punkt 2 linii 1\n" ;
		}	
	}
	;
	
	public abstract int size();
	public abstract String getHelp();
}
