package pl.struzek.msketch;

public class Test1 {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		

		Line ln1 = new Line(new Vector(1.0,1.0),new Vector(5.0,4.0));
		Circle c1 = new Circle(new Vector(3.0,0.0),new Vector(4.0,3.0));
		Circle c2 = new Circle(new Vector(3.0,0.0),new Vector(4.0,3.0));
		FixLine flc2 = new FixLine(new Vector(3.0,0.0),new Vector(4.0,3.0));
		FreePoint fp1 = new FreePoint(new Vector(1.0,1.0));
		System.out.println(ln1);
		System.out.println(c1);
		System.out.println(fp1);
		System.out.println(GeometricPrymitive.dbPrimitives);
	}

}
