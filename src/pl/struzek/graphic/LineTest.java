package pl.struzek.graphic;

import java.awt.geom.Line2D;

/**
 * Klasa reprezenujaca linie
 * mozna ja rysowac 
 * @author root
 *
 */
public class LineTest extends Line2D.Double {

	/** maksymalna odleglosc od linii */
	public static double minDist = 4;
	/** zaznaczony */
	boolean selected = false;
	
	public LineTest(double d, double e, double f, double g) {
		super(e, e, f, g);
	}

	@Override
	public boolean contains(double x, double y) {
		if(this.ptLineDist(x, y)<minDist){
			double x1,y1,x2,y2;
			double d,d1,d2;//dlugosci lini
			x1 = this.getX1();	x2 = this.getX2();	y1 = this.getY1();	y2 = this.getY2();
			
			d = Math.sqrt((x2-x1)*(x2-x1)+ (y2-y1)*(y2-y1));
			d1 = Math.sqrt((x1-x)*(x1-x)+ (y1-y)*(y1-y));
			d2 = Math.sqrt((x2-x)*(x2-x)+ (y2-y)*(y2-y));
			if((d1<=d) && (d2<=d)) return true;
		}
		return false;
	}
	
	public boolean isSelected() {
		return selected;
	}

	public void setSelected(boolean selected) {
		this.selected = selected;
	}

	public static void main(String[] args) {
		LineTest ln = new LineTest(0.0 , 0.0 , 10.0 ,0.0);
		System.out.println(ln.ptLineDist(5.0, 1.0));
		System.out.println(ln.contains(0.1, 1.0));
		System.out.println("Hello");
	}
}
