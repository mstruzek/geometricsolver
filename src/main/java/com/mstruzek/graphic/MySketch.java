package com.mstruzek.graphic;

import com.mstruzek.controller.ActiveKey;
import com.mstruzek.controller.Controller;
import com.mstruzek.msketch.Consts;
import com.mstruzek.msketch.GeometricPrimitive;
import com.mstruzek.msketch.Vector;

import javax.swing.*;
import javax.swing.event.MouseInputListener;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.geom.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.function.Predicate;

import static com.mstruzek.msketch.Point.getDbPoint;


public class MySketch extends JPanel implements MouseInputListener {
    /**
     *
     */
    private static final long serialVersionUID = 1L;
    public static final int S_WIDTH = 920;
    public static final int S_HEIGHT = 1000;
    public static final Color CONTROL_GUIDES_COLOR = new Color(57, 172, 115);
    public static final Color SYSTEM_COORDINATE_OX_COLOR = Color.RED;
    public static final Color SYSTEM_COORDINATE_OY_COLOR = Color.GREEN;
    public static final Color PANEL_BORDER_COLOR = Color.DARK_GRAY;
    public static final Color SKETCH_PANE_BACKGROUND_COLOR = new Color(160, 192, 222);

    ///tu skladujemy oddzielnie kategorie obiektow do odrysowania
    private final ArrayList<MyLine> lineStore = new ArrayList<>();
    private final ArrayList<MyCircle> circleStore = new ArrayList<>();
    private final ArrayList<MyFreePoint> freePointStore = new ArrayList<>();
    private final ArrayList<MyArc> arcStore = new ArrayList<>();
    private final ArrayList<MyPoint> pointStore = new ArrayList<>();

    /**
     * aktualny stan podczas klikania myszka
     */
    private MySketchState stateSketch = MySketchState.Normal;

    private final AffineTransform tx = new AffineTransform();

    private Point registeredPressed = new Point();

    final int r = 10;

    final MyPointContainer mpc = new MyPointContainer(-1, -1, -1, -1);

    private ActiveKey ack = ActiveKey.None;

    private final MyPopup popoup;

    /// Picked candidates for new geometric primitives !
    private int pId = 0;
    private Point[] picked = new Point[3];


    /**
     * glowny controller
     */
    private final Controller controller;

    private boolean controlGuidelines = false;

    public MySketch(Controller controller) {
        super();
        this.controller = controller;
        setLayout(null);
        setPreferredSize(new Dimension(S_WIDTH, S_HEIGHT));
        setBackground(SKETCH_PANE_BACKGROUND_COLOR);
        tx.translate(40, 900);
        tx.scale(1, -1);
        setBorder(BorderFactory.createTitledBorder(BorderFactory.createLineBorder(PANEL_BORDER_COLOR), "Szkicownik"));
        addMouseListener(this);
        addMouseMotionListener(this);
        this.popoup = new MyPopup(-1, mpc);
        this.popoup.addPropertyChangeListener(new PropertyChangeListener() {
            @Override
            public void propertyChange(PropertyChangeEvent evt) {
                /// forward outside
                firePropertyChange(evt.getPropertyName(), evt.getOldValue(), evt.getNewValue());
            }
        });
    }

    /**
     * Czy rysowanie linii kontrolnych wlaczone.
     */
    public boolean isControlGuidelines() {
        return controlGuidelines;
    }

    /**
     * Ustaw rysowanie linii kontrolnych.
     */
    public void setControlGuidelines(boolean guideLines) {
        this.controlGuidelines = guideLines;
    }

    public void setStateSketch(MySketchState stateSketch) {
        this.stateSketch = stateSketch;
    }

    /**
     * Pobierz dane z modelu
     */
    public void refreshContainers() {

        ArrayList<GeometricPrimitive> primitives = new ArrayList<>(GeometricPrimitive.dbPrimitives.values());

        MyPoint p1, p2, p3;
        MyLine ml;
        MyArc ma;
        MyCircle mc;
        MyFreePoint fp;

        lineStore.clear();
        circleStore.clear();
        freePointStore.clear();
        arcStore.clear();
        pointStore.clear();

        for (int i = 0; i < primitives.size(); i++) {
            GeometricPrimitive gm = primitives.get(i);

            switch (gm.getType()) {
                case Line:
                    p1 = new MyPoint(gm.getP1());
                    p2 = new MyPoint(gm.getP2());
                    add(p1);
                    add(p2);
                    ml = new MyLine(p1, p2);
                    ml.setPrimitiveId(gm.getPrimitiveId());
                    add(ml);
                    break;
                case Circle:
                    p1 = new MyPoint(gm.getP1());
                    p2 = new MyPoint(gm.getP2());
                    add(p1);
                    add(p2);
                    mc = new MyCircle(p1, p2);
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
                    ma = new MyArc(p1, p2, p3);
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
    }

    private void add(MyLine ml) {
        lineStore.add(ml);
    }

    private void add(MyCircle ml) {
        circleStore.add(ml);
    }

    private void add(MyFreePoint fp) {
        freePointStore.add(fp);
    }

    private void add(MyArc arc) {
        arcStore.add(arc);
    }

    private void add(MyPoint p) {
        pointStore.add(p);
    }

    @Override
    public void paint(Graphics g) {
        //promien punktu
        Graphics2D g2d = (Graphics2D) g;

        super.paint(g2d);

        Point tp1 = new Point();
        Point tp2 = new Point();
        Point tp3 = new Point();

        for (int i = 0; i < lineStore.size(); i++) {
            MyLine ml = lineStore.get(i);
            //transformacja
            tx.transform(ml.p1.getLocation(), tp1);
            tx.transform(ml.p2.getLocation(), tp2);
            g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));

            //dodatki
            if (controlGuidelines) {
                g2d.setColor(CONTROL_GUIDES_COLOR);

                int pA = GeometricPrimitive.dbPrimitives.get(ml.getPrimitiveId()).getA();
                int pB = GeometricPrimitive.dbPrimitives.get(ml.getPrimitiveId()).getB();
                //A - p1
                tp3.setLocation(getDbPoint().get(pA).getX(), getDbPoint().get(pA).getY());
                tx.transform(tp3, tp2);

                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));
                //B -p2
                tx.transform(ml.p2.getLocation(), tp2);
                tp3.setLocation(getDbPoint().get(pB).getX(), getDbPoint().get(pB).getY());
                tx.transform(tp3, tp1);
                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));

                g2d.setColor(PANEL_BORDER_COLOR);
            }

        }
        for (int i = 0; i < circleStore.size(); i++) {
            MyCircle cl = circleStore.get(i);
            //transformacja
            //System.out.println(cl.p1.getLocation());
            tx.transform(cl.p1.getLocation(), tp1);
            tx.transform(cl.p2.getLocation(), tp2);
            double r = Math.floor(tp1.distance(tp2));
            g2d.draw(new Ellipse2D.Double(tp1.x - r, tp1.y - r, 2 * r, 2 * r));
            g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));

            if (controlGuidelines) {
                g2d.setColor(CONTROL_GUIDES_COLOR);

                int pA = GeometricPrimitive.dbPrimitives.get(cl.getPrimitiveId()).getA();
                int pB = GeometricPrimitive.dbPrimitives.get(cl.getPrimitiveId()).getB();
                //A - p1
                tp3.setLocation(getDbPoint().get(pA).getX(), getDbPoint().get(pA).getY());
                tx.transform(tp3, tp2);

                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));
                //B -p2
                tx.transform(cl.p2.getLocation(), tp2);
                tp3.setLocation(getDbPoint().get(pB).getX(), getDbPoint().get(pB).getY());
                tx.transform(tp3, tp1);
                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));

                g2d.setColor(PANEL_BORDER_COLOR);
            }

        }
        for (int i = 0; i < arcStore.size(); i++) {

            MyArc cl = arcStore.get(i);
            //transformacja
            //System.out.println(cl.p1.getLocation());
            tx.transform(cl.p1.getLocation(), tp1);
            tx.transform(cl.p2.getLocation(), tp2);
            tx.transform(cl.p3.getLocation(), tp3);
            double r = Math.floor(tp1.distance(tp2));
            Arc2D.Double ark = new Arc2D.Double();
            //ark.setArcByTangent(tp2,tp1, tp3, r);
            Point v2 = new Point(tp2.x - tp1.x, tp2.y - tp1.y);
            Point v3 = new Point(tp3.x - tp1.x, tp3.y - tp1.y);
            double angSt = Consts.atan2(v2.y, v2.x);
            double angExt = Consts.atan2(v3.y, v3.x);
            double angDet = angExt - angSt;
            /*
             * Zabiegy potrzebne w celu zapewnienia
             * prawoskretnego ruchu wektor�w od Start -> end
             */

            if (angSt > angExt) {
                angDet = Math.PI * 2 + angDet;
            }

            //System.out.println(angSt*180/Math.PI + " , " + angExt*180/Math.PI + " , " + angDet*180/Math.PI);

            ark.setArcByCenter(tp1.x, tp1.y, r, angSt * 180 / Math.PI, angDet * 180 / Math.PI, Arc2D.OPEN);
            g2d.draw(ark);
            g2d.setColor(Color.getHSBColor(0.5f, 0.5f, 0.7f)); //pomaranczowym zaznaczam startowy kat
            g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));
            g2d.setColor(PANEL_BORDER_COLOR);
            g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp3.x, tp3.y));


            if (controlGuidelines) {
                g2d.setColor(CONTROL_GUIDES_COLOR);

                int pA = GeometricPrimitive.dbPrimitives.get(cl.getPrimitiveId()).getA();
                int pB = GeometricPrimitive.dbPrimitives.get(cl.getPrimitiveId()).getB();
                int pC = GeometricPrimitive.dbPrimitives.get(cl.getPrimitiveId()).getC();
                int pD = GeometricPrimitive.dbPrimitives.get(cl.getPrimitiveId()).getD();
                //A - p1
                tx.transform(cl.p1.getLocation(), tp1);
                tp3.setLocation(getDbPoint().get(pA).getX(), getDbPoint().get(pA).getY());
                tx.transform(tp3, tp2);
                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));

                //B -p1
                tx.transform(cl.p1.getLocation(), tp1);
                tp3.setLocation(getDbPoint().get(pB).getX(), getDbPoint().get(pB).getY());
                tx.transform(tp3, tp2);
                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));

                //C - p2
                tx.transform(cl.p2.getLocation(), tp2);
                tp3.setLocation(getDbPoint().get(pC).getX(), getDbPoint().get(pC).getY());
                tx.transform(tp3, tp1);
                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));
                //D - p3
                tx.transform(cl.p3.getLocation(), tp2);
                tp3.setLocation(getDbPoint().get(pD).getX(), getDbPoint().get(pD).getY());
                tx.transform(tp3, tp1);
                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));

                g2d.setColor(PANEL_BORDER_COLOR);
            }
            ;

        }
        for (int i = 0; i < freePointStore.size(); i++) {
            if (controlGuidelines) {
                g2d.setColor(CONTROL_GUIDES_COLOR);
                MyFreePoint mfp = freePointStore.get(i);

                int pA = GeometricPrimitive.dbPrimitives.get(mfp.getPrimitiveId()).getA();
                int pB = GeometricPrimitive.dbPrimitives.get(mfp.getPrimitiveId()).getB();
                //A - p1

                tx.transform(mfp.p1.getLocation(), tp1);
                tp3.setLocation(getDbPoint().get(pA).getX(), getDbPoint().get(pA).getY());
                tx.transform(tp3, tp2);

                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));
                //B -p1
                tx.transform(mfp.p1.getLocation(), tp1);
                tp3.setLocation(getDbPoint().get(pB).getX(), getDbPoint().get(pB).getY());
                tx.transform(tp3, tp2);
                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));

                g2d.setColor(PANEL_BORDER_COLOR);
            }
        }

        for (int i = 0; i < pointStore.size(); i++) {

            /*
            if (pointStore.get(i).isMouseClicked) {
                g2d.setColor(Color.RED);
            } else {
                g2d.setColor(Color.BLACK);
            }
*/

            tx.transform(pointStore.get(i).getLocation(), tp1);
            g2d.fill(new Ellipse2D.Double(tp1.x - r / 2, tp1.y - r / 2, r, r));
        }

        //Rysujemy osie
        Point p0 = new Point(0, 0);
        Point px = new Point(0, (int) Math.round(200 / tx.getScaleX()));
        Point py = new Point((int) Math.round(-200 / tx.getScaleY()), 0);
        tx.transform(px, px);
        tx.transform(py, py);
        tx.transform(p0, p0);
        g2d.setColor(SYSTEM_COORDINATE_OY_COLOR);
        g2d.draw(new Line2D.Double(p0, px)); //X
        g2d.setColor(SYSTEM_COORDINATE_OX_COLOR);
        g2d.draw(new Line2D.Double(p0, py)); //Y
    }


    public Point getInverseTransformEvent(MouseEvent e) {
        Point tp1 = new Point();
        try {
            tx.inverseTransform(e.getPoint().getLocation(), tp1);
        } catch (NoninvertibleTransformException e1) {
            e1.printStackTrace();
        }
        return tp1;
    }
    //INTERFEJS MOZNA WYKORZYSTAC DO SKALOWANIA WSZYSTKICH PUNKTOW


    /**
     * Zachowanie Keybord and Mouse :
     * <p>
     * - Keyboard ( button Shift ) + Click( green space)    => zeruj buffor KLMN
     * <p>
     * - Keybord ( button Ctrl ) + Click( point)            => wypelnij dwa nastepne wolne sloty bufforze KLMN
     * <p>
     * - Keybord ( button K ) ... Click( point )            => umiesc w burze K
     * <p>
     * - Keybord ( button L ) ... Click( point )            => umiesc w burze L
     * <p>
     * - Keybord ( button M ) ... Click( point )            => umiesc w burze M
     * <p>
     * - Keybord ( button N ) ... Click( point )            => umiesc w burze N
     *
     * @param e the event to be processed
     */
    @Override
    public void mouseClicked(MouseEvent e) {
        //pierw sprawdzamy czy klikniecie w jeden z naszych punktow
        Point tp1 = getInverseTransformEvent(e);

        boolean found = false;
        //System.out.println(tp1);
        for (int i = 0; i < pointStore.size(); i++) {
            MyPoint viewPoint = pointStore.get(i);
            if (viewPoint.contains(tp1.x, tp1.y, 2 * r / tx.getScaleX())) {
                if (ack == ActiveKey.K) {
                    ack = ActiveKey.None;
                    mpc.setPointK(viewPoint.id);
                } else if (ack == ActiveKey.L) {
                    ack = ActiveKey.None;
                    mpc.setPointL(viewPoint.id);
                } else if (ack == ActiveKey.M) {
                    ack = ActiveKey.None;
                    mpc.setPointM(viewPoint.id);
                } else if (ack == ActiveKey.N) {
                    ack = ActiveKey.None;
                    mpc.setPointN(viewPoint.id);
                } else if (ack == ActiveKey.None && e.isControlDown()) {

                    int first = viewPoint.id;

                    GeometricPrimitive parentPrimitive =
                        GeometricPrimitive.dbPrimitives.values().stream()
                            .filter(p -> Arrays.stream(p.getAllPointsId()).boxed().anyMatch(Predicate.isEqual(first)))
                            .findFirst()
                            .orElseThrow(() -> new IndexOutOfBoundsException("point id :" + first));

                    // set two consecutive free slots
                    switch (parentPrimitive.getType()) {
                        case FreePoint:
                            mpc.setFreeSlot(parentPrimitive.getP1());
                            break;
                        case Arc:
                        case Line:
                            if (parentPrimitive.getP1() == first) {
                                mpc.setFreeSlot(first);                     // K
                                mpc.setFreeSlot(parentPrimitive.getP2());   // L
                            } else if (parentPrimitive.getP2() == first) {
                                mpc.setFreeSlot(first);                     // K
                                mpc.setFreeSlot(parentPrimitive.getP1());   // L
                            }
                            break;
                        case Circle:
                            mpc.setFreeSlot(parentPrimitive.getP1()); // M
                            mpc.setFreeSlot(parentPrimitive.getP2()); // N
                            break;
                        case FixLine:
                            // control OXY lines
                            break;
                    }
                } else {
                    firePropertyChange(Property.SELECTED_POINT, null, com.mstruzek.msketch.Point.dbPoint.get(viewPoint.id).toString());
                }
                firePropertyChange(Property.KLMN_POINTS, null, mpc.toString());
                found = true;
                break;
            }
        }
        if (!found && e.isShiftDown()) {
            mpc.clearAll();
            firePropertyChange(Property.KLMN_POINTS, null, mpc.toString());
        }
        repaint();
    }

    @Override
    public void mouseEntered(MouseEvent e) {

    }

    @Override
    public void mouseExited(MouseEvent e) {

    }

    private MyPoint pressedPoint = null;

    @Override
    public void mousePressed(MouseEvent e) {

        //odnosnie skalowania
        if (e.isControlDown()) {
            registeredPressed = e.getPoint();
            repaint();
            return;
        }

        //odnosnie przemieszczania przesuwania
        if (e.isShiftDown()) {
            registeredPressed = e.getPoint();
            repaint();
            return;
        }

        //pierw sprawdzamy czy klikniecie w jeden z naszych punktow
        Point tp1 = getInverseTransformEvent(e);

        for (int i = 0; i < pointStore.size(); i++) {
            if (pointStore.get(i).contains(tp1.x, tp1.y, r / tx.getScaleX())) {
                pressedPoint = pointStore.get(i);
                pressedPoint.setDragged(true);
                break;
            }
        }
        repaint();
    }


    @Override
    public void mouseReleased(MouseEvent e) {
        //pierw sprawdzamy czy klikniecie w jeden z naszych punktow
        Point tp1 = getInverseTransformEvent(e);

        switch (stateSketch) {

            case Normal:
                if (pressedPoint != null) {
                    MyPoint selected = pressedPoint;
                    pressedPoint.setDragged(false);
                    pressedPoint = null;
                    if (e.isPopupTrigger()) {
                        popoup.setPointId(selected.id);
                        popoup.show(e.getComponent(), e.getX(), e.getY());
                    }
                }
                break;

            case DrawLine:
                //oczekuj 2 klikniec
                if (picked[pId] == null) {
                    picked[pId++] = new Point(tp1);
                    if (pId == 2) {
                        Point p1 = picked[0];
                        Point p2 = picked[1];
                        controller.addLine(new Vector(p1.x, p1.y), new Vector(p2.x, p2.y));
                        refreshContainers();
                        stateSketch = MySketchState.Normal;
                        pId = 0;
                        picked[0] = null;
                        picked[1] = null;
                        repaint();
                    }
                }
                return;

            case DrawCircle:
                //oczekuje 2 klikniec
                if (picked[pId] == null) {
                    picked[pId++] = new Point(tp1);
                    if (pId == 2) {
                        Point p1 = picked[0];
                        Point p2 = picked[1];
                        controller.addCircle(new Vector(p1.x, p1.y), new Vector(p2.x, p2.y));
                        refreshContainers();
                        stateSketch = MySketchState.Normal;
                        pId = 0;
                        picked[0] = null;
                        picked[1] = null;
                        repaint();
                    }
                }
                return;

            case DrawArc:
                if (picked[pId] == null) {
                    picked[pId++] = new Point(tp1);
                    //oczekuje 3 klikniec
                    if (pId == 3) {
                        Point p1 = picked[0];
                        Point p2 = picked[1];
                        Point p3 = picked[2];
                        controller.addArc(new Vector(p1.x, p1.y), new Vector(p2.x, p2.y), new Vector(p3.x, p3.y));
                        refreshContainers();
                        stateSketch = MySketchState.Normal;
                        pId = 0;
                        picked[0] = null;
                        picked[1] = null;
                        picked[2] = null;
                        repaint();
                    }
                }
                return;

            case DrawPoint:
                if (picked[pId] == null) {
                    picked[pId++] = new Point(tp1);
                    //oczekuje 3 klikniec
                    if (pId == 1) {
                        Point p1 = picked[0];
                        controller.addPoint(new Vector(p1.x, p1.y));
                        refreshContainers();
                        stateSketch = MySketchState.Normal;
                        pId = 0;
                        picked[0] = null;
                        repaint();
                    }
                }
                return;
        }
    }

    @Override
    public void mouseDragged(MouseEvent e) {

        if (e.isControlDown()) {
            double k = e.getPoint().getY() - registeredPressed.getY();
            tx.scale(k / 100 + 1, k / 100 + 1);
            registeredPressed = e.getPoint();
            repaint();
            return;
        }
        //odnosnie przesuwania
        if (e.isShiftDown()) {
            Point tr = new Point((int) Math.round(e.getPoint().getX() - registeredPressed.getX()), (int) Math.round(e.getPoint().getY() - registeredPressed.getY()));
            tx.translate(tr.getX() / tx.getScaleX(), tr.getY() / tx.getScaleY());
            registeredPressed = e.getPoint();
            repaint();
            return;
        }
        //pierw sprawdzamy czy klikniecie w jeden z naszych punktow
        Point point = getInverseTransformEvent(e);

        if (pressedPoint != null) {
            Point p = new Point((int) (point.getX()), (int) (point.getY()));
            pressedPoint.setLocation(p);
        }
        repaint();
    }

    @Override
    public void mouseMoved(MouseEvent e) {
        Point tp1 = new Point();
        try {
            tx.inverseTransform(e.getPoint().getLocation(), tp1);
        } catch (NoninvertibleTransformException e1) {
            e1.printStackTrace();
        }
        firePropertyChange(Property.CURRENT_POSITION, null, String.format("X : %1$5.3f , Y : %2$5.3f", tp1.getX(), tp1.getY()));
    }

    public int getPointK() {
        return mpc.getPointK();
    }

    public int getPointL() {
        return mpc.getPointL();
    }

    public int getPointM() {
        return mpc.getPointM();
    }

    public int getPointN() {
        return mpc.getPointN();
    }

    public void clearAll() {
        mpc.clearAll();
    }

    public void setAck(ActiveKey activeKey) {
        this.ack = activeKey;
    }
}

