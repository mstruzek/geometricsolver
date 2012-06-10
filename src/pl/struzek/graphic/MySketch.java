package pl.struzek.graphic;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.geom.AffineTransform;
import java.awt.geom.Arc2D;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.geom.NoninvertibleTransformException;

import java.util.ArrayList;

import javax.swing.BorderFactory;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.event.MouseInputListener;

import pl.struzek.controller.Controller;
import pl.struzek.msketch.Config;
import pl.struzek.msketch.GeometricPrymitive;
import pl.struzek.msketch.Vector;


public class MySketch extends JPanel implements MouseInputListener ,KeyListener{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	//tu skladujemy linie
	ArrayList<MyLine> lineStore = new ArrayList<MyLine>();
	ArrayList<MyCircle> circleStore = new ArrayList<MyCircle>();
	ArrayList<MyFreePoint> freePointStore = new ArrayList<MyFreePoint>();
	ArrayList<MyArc> arcStore = new ArrayList<MyArc>();
	ArrayList<MyPoint> pointStore = new ArrayList<MyPoint>();
	
	/** aktualny stan podczas klikania myszka */
	MySketchState stateSketch = MySketchState.Normal;
	
	Point lastPosition = new Point();
	
	AffineTransform tx = new AffineTransform();
	
	int r =10;
	
	public MyPointContainer mpc = new MyPointContainer(-1,-1,-1,-1);
	
	public JLabel jl =new JLabel("K,L,M,N");	
	
	public JLabel currentPosition = new JLabel("Currrent Position:");
	MyPopup popoup = new MyPopup(-1,mpc,jl);
	

	
	Point tmp1 = null;
	
	/** pomin tyle punktow podczas wybierania prawym przyciskiem */
	int pominPunkty;
	
	/** glowny controller */
	Controller controller;
	
	boolean withControlLines =	false;
	
	public MySketch(Controller controller){
		super();
		this.controller = controller;
		setLayout(null);
		tx.translate(20, 300);
		tx.scale(1, -1);
		
		setBorder(BorderFactory.createTitledBorder("Szkicownik"));
			
		addMouseListener(this);
		addMouseMotionListener(this);

	}

	/** Czy rysowanie linii kontrolnych wlaczone */
	public boolean isWithControlLines() {
		return withControlLines;
	}
	
	/** Ustaw rysowanie linii kontrolnych */
	public void setWithControlLines(boolean withControlLines) {
		this.withControlLines = withControlLines;
	}

	public void add(MyLine ml){
		lineStore.add(ml);
		//pointStore.add(ml.getPoints()[0]);
		//pointStore.add(ml.getPoints()[1]);
	}
	
	public void add(MyCircle ml){
		circleStore.add(ml);
		//lineStore.add(new MyLine(ml.p1,ml.p2));
		//pointStore.add(ml.getPoints()[0]);
		//pointStore.add(ml.getPoints()[1]);
	}
	public void add(MyFreePoint fp){
		freePointStore.add(fp);
	}	
	public void add(MyArc arc){
		arcStore.add(arc);
		//lineStore.add(new MyLine(arc.p1,arc.p2));
		//lineStore.add(new MyLine(arc.p1,arc.p3));
	}
	public void add(MyPoint p){
		pointStore.add(p);
	}
	
	public MySketchState getStateSketch() {
		return stateSketch;
	}

	public void setStateSketch(MySketchState stateSketch) {
		this.stateSketch = stateSketch;
	}
	
	/** Pobierz dane z modelu */
	public void refreshContainers(){
		ArrayList<GeometricPrymitive> primitives= controller.getPrimitivesContainer();
		
		MyPoint p1,p2,p3;
		MyLine ml;
		MyPoint mp;
		MyArc ma;
		MyCircle mc;
		MyFreePoint fp;
		
		lineStore.clear();
		circleStore.clear();
		freePointStore.clear();
		arcStore.clear();
		pointStore.clear();
		
		for(int i=0;i<primitives.size();i++){
			GeometricPrymitive gm = primitives.get(i);

			switch(gm.getType()){
			case Line:
				p1 = new MyPoint(gm.getP1());
				p2 = new MyPoint(gm.getP2());
				add(p1);
				add(p2);
				ml = new MyLine(p1,p2);
				ml.setPrimitiveId(gm.getPrimitiveId());
				add(ml);
				break;
			case Circle:
				p1 = new MyPoint(gm.getP1());
				p2 = new MyPoint(gm.getP2());
				add(p1);
				add(p2);
				mc = new MyCircle(p1,p2);
				mc.setPrimitiveId(gm.getPrimitiveId());
				add(mc);				
				break;
			case Arc:
				p1 = new MyPoint(gm.getP1());
				p2 = new MyPoint(gm.getP2());
				p3 = new MyPoint(gm.getP3());
				add(p1);
				add(p2);
				add(p3);
				ma = new MyArc(p1,p2,p3);
				ma.setPrimitiveId(gm.getPrimitiveId());
				add(ma);				
				break;				
			case FreePoint:
				p1 = new MyPoint(gm.getP1());
				add(p1);
				fp = new MyFreePoint(p1);
				fp.setPrimitiveId(gm.getPrimitiveId());
				add(fp);
				break;
			}
		}
		repaint();
	}

	@Override
	public void paint(Graphics g) {
		//promien punktu
		Graphics2D g2d = (Graphics2D)g;

		super.paint(g2d);	
		
		Point tp1= new Point();
		Point tp2 = new Point();
		Point tp3 = new Point();
		
		for(int i=0;i<lineStore.size();i++){
			MyLine ml = lineStore.get(i);
			//transformacja
			tx.transform(ml.p1.getLocation(), tp1);
			tx.transform(ml.p2.getLocation(), tp2);
			g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x,  tp2.y));
			
			//dodatki
			if(withControlLines){
				g2d.setColor(Color.LIGHT_GRAY);
				
				int pA= GeometricPrymitive.dbPrimitives.get(ml.getPrimitiveId()).getA();
				int pB= GeometricPrymitive.dbPrimitives.get(ml.getPrimitiveId()).getB();
				//A - p1
				tp3.setLocation(pl.struzek.msketch.Point.getDbPoint().get(pA).getX(), pl.struzek.msketch.Point.getDbPoint().get(pA).getY());
				tx.transform(tp3, tp2);
				
				g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x,  tp2.y));
				//B -p2
				tx.transform(ml.p2.getLocation(), tp2);
				tp3.setLocation(pl.struzek.msketch.Point.getDbPoint().get(pB).getX(), pl.struzek.msketch.Point.getDbPoint().get(pB).getY());
				tx.transform(tp3, tp1);
				g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x,  tp2.y));
				
				g2d.setColor(Color.BLACK);				
			}

		}
		for(int i=0;i<circleStore.size();i++){
			MyCircle cl = circleStore.get(i);
			//transformacja
			//System.out.println(cl.p1.getLocation());
			tx.transform(cl.p1.getLocation(), tp1);
			tx.transform(cl.p2.getLocation(), tp2);
			double r = Math.floor(tp1.distance(tp2));
			g2d.draw(new Ellipse2D.Double(tp1.x-r,tp1.y-r,2*r,2*r));
			g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x,  tp2.y));
			
			if(withControlLines){
				g2d.setColor(Color.LIGHT_GRAY);
				
				int pA= GeometricPrymitive.dbPrimitives.get(cl.getPrimitiveId()).getA();
				int pB= GeometricPrymitive.dbPrimitives.get(cl.getPrimitiveId()).getB();
				//A - p1
				tp3.setLocation(pl.struzek.msketch.Point.getDbPoint().get(pA).getX(), pl.struzek.msketch.Point.getDbPoint().get(pA).getY());
				tx.transform(tp3, tp2);
				
				g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x,  tp2.y));
				//B -p2
				tx.transform(cl.p2.getLocation(), tp2);
				tp3.setLocation(pl.struzek.msketch.Point.getDbPoint().get(pB).getX(), pl.struzek.msketch.Point.getDbPoint().get(pB).getY());
				tx.transform(tp3, tp1);
				g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x,  tp2.y));
				
				g2d.setColor(Color.BLACK);				
			};
			
		}
		for(int i=0;i<arcStore.size();i++){
			
			MyArc cl = arcStore.get(i);
			//transformacja
			//System.out.println(cl.p1.getLocation());
			tx.transform(cl.p1.getLocation(), tp1);
			tx.transform(cl.p2.getLocation(), tp2);
			tx.transform(cl.p3.getLocation(), tp3);
			double r = Math.floor(tp1.distance(tp2));
			Arc2D.Double ark = new Arc2D.Double();
			//ark.setArcByTangent(tp2,tp1, tp3, r);
			Point v2 =new Point(tp2.x-tp1.x,tp2.y-tp1.y); 
			Point v3 =new Point(tp3.x-tp1.x,tp3.y-tp1.y); 
			double angSt = Config.atan2(v2.y, v2.x);
			double angExt = Config.atan2(v3.y, v3.x);
			double angDet = angExt-angSt;
			/*
			 * Zabiegy potrzebne w celu zapewnienia 
			 * prawoskretnego ruchu wektorów od Start -> end
			 */
			
			if(angSt>angExt){
				angDet = Math.PI*2+angDet;
			}
			
			//System.out.println(angSt*180/Math.PI + " , " + angExt*180/Math.PI + " , " + angDet*180/Math.PI);
			
			ark.setArcByCenter(tp1.x, tp1.y, r, angSt*180/Math.PI, angDet*180/Math.PI ,Arc2D.OPEN);
			g2d.draw(ark);
			g2d.setColor(Color.getHSBColor(0.5f, 0.5f, 0.7f)); //pomaranczowym zaznaczam startowy kat
			g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x,  tp2.y));
			g2d.setColor(Color.BLACK);
			g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp3.x,  tp3.y));
			
			
			if(withControlLines){
				g2d.setColor(Color.LIGHT_GRAY);
				
				int pA= GeometricPrymitive.dbPrimitives.get(cl.getPrimitiveId()).getA();
				int pB= GeometricPrymitive.dbPrimitives.get(cl.getPrimitiveId()).getB();
				int pC= GeometricPrymitive.dbPrimitives.get(cl.getPrimitiveId()).getC();
				int pD= GeometricPrymitive.dbPrimitives.get(cl.getPrimitiveId()).getD();
				//A - p1
				tx.transform(cl.p1.getLocation(), tp1);
				tp3.setLocation(pl.struzek.msketch.Point.getDbPoint().get(pA).getX(), pl.struzek.msketch.Point.getDbPoint().get(pA).getY());
				tx.transform(tp3, tp2);
				g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x,  tp2.y));
				
				//B -p1
				tx.transform(cl.p1.getLocation(), tp1);
				tp3.setLocation(pl.struzek.msketch.Point.getDbPoint().get(pB).getX(), pl.struzek.msketch.Point.getDbPoint().get(pB).getY());
				tx.transform(tp3, tp2);
				g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x,  tp2.y));
				
				//C - p2
				tx.transform(cl.p2.getLocation(), tp2);
				tp3.setLocation(pl.struzek.msketch.Point.getDbPoint().get(pC).getX(), pl.struzek.msketch.Point.getDbPoint().get(pC).getY());
				tx.transform(tp3, tp1);
				g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x,  tp2.y));
				//D - p3
				tx.transform(cl.p3.getLocation(), tp2);
				tp3.setLocation(pl.struzek.msketch.Point.getDbPoint().get(pD).getX(), pl.struzek.msketch.Point.getDbPoint().get(pD).getY());
				tx.transform(tp3, tp1);
				g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x,  tp2.y));	
				
				g2d.setColor(Color.BLACK);				
			};
			
		}		
		for(int i=0;i<freePointStore.size();i++){
			if(withControlLines){
				g2d.setColor(Color.LIGHT_GRAY);
				MyFreePoint mfp = freePointStore.get(i);
				
				int pA= GeometricPrymitive.dbPrimitives.get(mfp.getPrimitiveId()).getA();
				int pB= GeometricPrymitive.dbPrimitives.get(mfp.getPrimitiveId()).getB();
				//A - p1
				
				tx.transform(mfp.p1.getLocation(), tp1);
				tp3.setLocation(pl.struzek.msketch.Point.getDbPoint().get(pA).getX(), pl.struzek.msketch.Point.getDbPoint().get(pA).getY());
				tx.transform(tp3, tp2);
				
				g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x,  tp2.y));
				//B -p1
				tx.transform(mfp.p1.getLocation(), tp1);
				tp3.setLocation(pl.struzek.msketch.Point.getDbPoint().get(pB).getX(), pl.struzek.msketch.Point.getDbPoint().get(pB).getY());
				tx.transform(tp3, tp2);
				g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x,  tp2.y));
				
				g2d.setColor(Color.BLACK);		
			}
		}
		
		for(int i=0;i<pointStore.size();i++){
			if(pointStore.get(i).isMouseClicked){
				g2d.setColor( Color.RED);
			}else{
				g2d.setColor( Color.BLACK);
			}

			tx.transform(pointStore.get(i).getLocation(), tp1);
			g2d.fill(new Ellipse2D.Double(tp1.x-r/2,tp1.y-r/2,r,r));
		}	
		
		//Rysujemy osie
		Point p0 =new Point(0,0);
		Point px = new Point(0,(int)Math.round(200/tx.getScaleX()));
		Point py = new Point((int)Math.round(-200/tx.getScaleY()),0);
		tx.transform(px	,px);
		tx.transform(py	,py);
		tx.transform(p0	,p0);
		g2d.setColor( Color.RED);
		g2d.draw(new Line2D.Double(p0,px)); //X
		g2d.draw(new Line2D.Double(p0,py)); //Y
	}


	public Point getInversTransformeEvent(MouseEvent e){
		Point tp1= new Point();
		try {
			tx.inverseTransform(e.getPoint().getLocation(),tp1);
		} catch (NoninvertibleTransformException e1) {
			e1.printStackTrace();
		}
		return tp1;		
	}
	//INTERFEJS MOZNA WYKORZYSTAC DO SKALOWANIA WSZYSTKICH PUNKTOW
	
	// Z kontrolem bedzie obsluga AffineTrnasform
	@Override
	public void mouseClicked(MouseEvent e) {
		//pierw sprawdzamy czy klikniecie w jeden z naszych punktow
		Point tp1= getInversTransformeEvent(e);
		
		//System.out.println(tp1);
		for(int i=0;i<pointStore.size();i++){
			if(pointStore.get(i).contains(tp1.x, tp1.y,r/tx.getScaleX())){
				if(pointStore.get(i).isMouseClicked){
					pointStore.get(i).isMouseClicked=false;
				}else{
					pointStore.get(i).isMouseClicked=true;
						

				}
				break;
			}
		}	
		repaint();		
	}

	@Override
	public void mouseEntered(MouseEvent e) {
		
	}

	@Override
	public void mouseExited(MouseEvent e) {
		
	}

	@Override
	public void mousePressed(MouseEvent e) {
		
		//odnosnie skalowania
		if(e.isControlDown()){
			lastPosition = e.getPoint();
			repaint();	
			return;
		}
		
		//odnosnie przemieszczania przesuwania
		if(e.isShiftDown()){
			lastPosition = e.getPoint();
			repaint();	
			return;			
		}
		//pierw sprawdzamy czy klikniecie w jeden z naszych punktow
		Point tp1= getInversTransformeEvent(e);
		
		//System.out.println(tp1);
		for(int i=0;i<pointStore.size();i++){
			if(pointStore.get(i).contains(tp1.x, tp1.y,r/tx.getScaleX())){
				pointStore.get(i).setDragged(true);
				break;
			}
		}	
		repaint();			
	}

	@Override
	public void mouseReleased(MouseEvent e) {
		//pierw sprawdzamy czy klikniecie w jeden z naszych punktow
		Point tp1= getInversTransformeEvent(e);
		
		switch(stateSketch){
		
		case Normal:

			for(int i=0;i<pointStore.size();i++){
				if(pointStore.get(i).contains(tp1.x,tp1.y,r/tx.getScaleX())){
					pointStore.get(i).setDragged(false);
					
					if(pominPunkty>0){
						pominPunkty--;
						continue;
					}

					//popup menu
					if(e.isPopupTrigger()){
						popoup.setPointId(pointStore.get(i).id);
						popoup.rebuild();
						popoup.show(e.getComponent(),e.getX(),e.getY());
					}
					break;
				}
			}		
		
			break;
			
		case DrawLine:
			//oczekuj 2 klikniec
			
			if(tmp1==null){
				
				tmp1 = new Point(tp1);
				
			}else{
				Point tmp2= new Point(tp1);
			
				//add(new MyLine(tmp1.x,tmp1.y,tmp2.x,tmp2.y));
				controller.addLine(new Vector(tmp1.x,tmp1.y), new Vector(tmp2.x,tmp2.y));
				
				tmp1 =null;
				refreshContainers();
				stateSketch = MySketchState.Normal;
			}
			
			break;
			
		case DrawCircle:
			//oczekuj 2 klikniec
			if(tmp1==null){
				
				tmp1 = new Point(tp1);
				
			}else{
				Point tmp2 = new Point(tp1);
				
				//add(new MyCircle(tmp1.x,tmp1.y,tmp2.x,tmp2.y));
				controller.addCircle(new Vector(tmp1.x,tmp1.y), new Vector(tmp2.x,tmp2.y));
				tmp1 =null;
				refreshContainers();
				stateSketch = MySketchState.Normal;
			}			
			break;
			
		case DrawArc:
			//oczekuj 2 klikniec
			if(tmp1==null){
				
				tmp1 = new Point(tp1);
				
			}else{
				Point tmp2 = new Point(tp1);
				
				//add(new MyCircle(tmp1.x,tmp1.y,tmp2.x,tmp2.y));
				controller.addArc(new Vector(tmp1.x,tmp1.y), new Vector(tmp2.x,tmp2.y));
				tmp1 =null;
				refreshContainers();
				stateSketch = MySketchState.Normal;
			}			
			break;			
		case DrawPoint:
			//oczekuj 1 klikniecie
				Point tmp2 = new Point(tp1);
				
				//add(new MyPoint(tmp2.x,tmp2.y));
				controller.addPoint(new Vector(tmp2.x,tmp2.y));
				tmp1 =null;
				refreshContainers();
				stateSketch = MySketchState.Normal;
	
			break;
		}

		//this.getParent().getParent().getParent().getParent().getParent().setFocusable(true);

		//repaint(); // jest w refreshContainer				
	}

	@Override
	public void mouseDragged(MouseEvent e) {
		
		if(e.isControlDown()){
			double k = e.getPoint().getY() - lastPosition.getY();
			//System.out.println(k);
			tx.scale(( 1+k/100), (1+k/100));	
			lastPosition=e.getPoint();
			//System.out.println(tx);
			repaint();	
			return;
		}
		//odnosnie przesuwania
		if(e.isShiftDown()){
			Point tr = new Point((int)Math.round(e.getPoint().getX()-lastPosition.getX()),(int)Math.round(e.getPoint().getY()-lastPosition.getY()));
			tx.translate(tr.getX()/tx.getScaleX(), tr.getY()/tx.getScaleY());
			lastPosition=e.getPoint();
			repaint();	
			return;			
		}
		//pierw sprawdzamy czy klikniecie w jeden z naszych punktow
		Point tp1= getInversTransformeEvent(e);

		for(int i=0;i<pointStore.size();i++){
			if(pointStore.get(i).isDragged()){
				Point p= new Point((int)(tp1.getX()),(int)( tp1.getY()));
				pointStore.get(i).setLocation(p);
				break;
			}
		}	
		repaint();					
	}

	@Override
	public void mouseMoved(MouseEvent e) {
		Point tp1 = new Point();
		try {
			tx.inverseTransform(e.getPoint().getLocation(),tp1);
		} catch (NoninvertibleTransformException e1) {
			e1.printStackTrace();
		}
		this.currentPosition.setText(String.format("X : %1$5.3f , Y : %2$5.3f", tp1.getX(),tp1.getY()));

	}

	@Override
	public void keyPressed(KeyEvent e) {	
	}

	@Override
	public void keyReleased(KeyEvent e) {	
		pominPunkty  = e.getKeyCode()-48;
		
		//System.out.println(e);
	}

	@Override
	public void keyTyped(KeyEvent e) {	

	}

	public KeyListener getMyKeyListener() {
		return this;
	}

	
}

