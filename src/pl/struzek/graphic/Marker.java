package pl.struzek.graphic;

import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Point2D;

/**
 * Klasa wyswietla dany Marke (Punkt) ,
 * na naszym szkicowniku
 * @author root
 *
 */
class Marker {
	
	/** promien */
    static final double radius = 3;
    /** okrag */
	Ellipse2D.Double circle;
	/** srodek okregu*/
	Point2D.Double center;

	public Marker(Point2D.Double control) {
      center = control; 
      circle = new Ellipse2D.Double(control.x - radius, control.y - radius, 2.0 * radius,
          2.0 * radius);
    }
	
    public boolean contains(double x, double y) {
	  return circle.contains(x, y);
	}
    
	public void setLocation(double x, double y) {
	  center.x = x; 
	  center.y = y; 
	  circle.x = x - radius; 
	  circle.y = y - radius; 
	}
	
	public Point2D.Double getCenter() {
	  return center;
	}
	
	public void draw(Graphics2D g2D) {
      g2D.draw(circle);
    }
  }

