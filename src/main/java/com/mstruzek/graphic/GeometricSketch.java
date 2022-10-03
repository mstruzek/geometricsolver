package com.mstruzek.graphic;

import com.mstruzek.controller.Controller;
import com.mstruzek.msketch.Consts;
import com.mstruzek.msketch.GeometricObject;
import com.mstruzek.msketch.ModelRegistry;
import com.mstruzek.msketch.Vector;

import javax.swing.*;
import javax.swing.event.MouseInputListener;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.geom.*;
import java.util.*;
import java.util.function.Predicate;

import static com.mstruzek.graphic.Property.KLMN_POINTS;
import static java.lang.System.currentTimeMillis;


public class GeometricSketch extends JPanel implements MouseInputListener, MouseWheelListener {
    private static final int S_WIDTH = 1120;
    private static final int S_HEIGHT = 900;
    public static final String TITLE_SZKICOWNIK = "Szkicownik";

    ///tu skladujemy oddzielnie kategorie obiektow do odrysowania
    private final ArrayList<DrawLine> lineContainer = new ArrayList<>();
    private final ArrayList<DrawCircle> circleContainer = new ArrayList<>();
    private final ArrayList<DrawFreePoint> freePointContainer = new ArrayList<>();
    private final ArrayList<DrawArc> arcContainer = new ArrayList<>();
    private final HashMap<Integer, DrawPoint> pointContainer = new HashMap<>();

    /// aktualny stan podczas klikania myszka
    private volatile ApplicationState state = ApplicationState.Normal;
    private final AffineTransform G = new AffineTransform();

    private volatile GeometricSOAIndex geometricSoAIndex = new GeometricSOAIndex();

    /// random gitter for NaN translation lower bound
    private static final Random jitterSource = new Random();
    private static final double JITTER_MAX = 1e-5;
    private static final double JITTER_MIN = 1e-4;

    /// Zoom-In / Zoom-Out with left Ctrl button
    private static final double CTRL_SCALE_SPD2 = 16.0;
    private static final double CTRL_SCALE_SPD1 = 8.0;
    private static final double SCALE_BASE = 100.0;

    // structure use for transformation of coordinate system
    private volatile Point lastPressed = null;
    /// hold last point id selected for drag operation
    private volatile Integer pressedId = null;

    private final PointContainer mpc = new PointContainer(-1, -1, -1, -1);
    private final PointSelectorPopup popoup;

    ///  main model controller
    private final Controller controller;

    private final int pRadius = 10;
    /// Picked candidates for new geometric primitives !
    private final Point[] picked = new Point[3];
    private int pId = 0;

    private static final long AWAIT_HOVER_TIMEOUT = 100; // [ ms ]
    private volatile long selectionTime = 0L;
    private volatile boolean hoverState = false;

    private static final Color CONTROL_GUIDES_COLOR = new Color(57, 172, 115);
    private static final Color SYSTEM_COORDINATE_OX_COLOR = Color.RED;
    private static final Color SYSTEM_COORDINATE_OY_COLOR = Color.GREEN;
    private static final Color PANEL_BORDER_COLOR = Color.DARK_GRAY;
    private static final Color SKETCH_PANE_BACKGROUND_COLOR = new Color(160, 192, 222);

    private boolean controlGuidelines = false;

    public GeometricSketch(Controller controller) {
        super();
        this.controller = controller;
        setLayout(null);
        setPreferredSize(new Dimension(S_WIDTH, S_HEIGHT));
        setBackground(SKETCH_PANE_BACKGROUND_COLOR);

        /// default coordinates translation, and Y scala
        G.scale(1.0, -1.0);
        G.translate(30.0, -800.0);

        setBorder(BorderFactory.createTitledBorder(BorderFactory.createLineBorder(PANEL_BORDER_COLOR), TITLE_SZKICOWNIK));
        addMouseListener(this);
        addMouseMotionListener(this);
        addMouseWheelListener(this);

        this.popoup = new PointSelectorPopup(-1, mpc);
        this.popoup.addPropertyChangeListener(evt ->
            firePropertyChange(evt.getPropertyName(), evt.getOldValue(), evt.getNewValue()));
    }

    /// Czy rysowanie linii kontrolnych wlaczone !
    public boolean isControlGuidelines() {
        return controlGuidelines;
    }

    /// Ustaw rysowanie linii kontrolnych.
    public void setControlGuidelines(boolean guideLines) {
        this.controlGuidelines = guideLines;
    }

    public void setState(ApplicationState state) {
        this.state = state;
    }


    /// Internally used after location changed
    private void updateGeometricSOAIndex() {
        GeometricSOAIndex geometricSoAIndex = new GeometricSOAIndex();
        for (DrawPoint point : pointContainer.values()) {
            Point p = point.getLocation();
            geometricSoAIndex.addGeometricPoint(point.id, p.x, p.y);

        }
        this.geometricSoAIndex = geometricSoAIndex;
    }

    public void rebuildViewModel() {
        final GeometricSOAIndex geometricSoAIndex = new GeometricSOAIndex();
        /// temporary evaluation objects
        DrawPoint p1, p2, p3;
        DrawLine ml;
        DrawArc ma;
        DrawCircle mc;
        DrawFreePoint fp;

        ///  temporary objects
        Point loc1, loc2, loc3;

        lineContainer.clear();
        circleContainer.clear();
        freePointContainer.clear();
        arcContainer.clear();
        pointContainer.clear();
        ArrayList<GeometricObject> primitives = new ArrayList<>(ModelRegistry.dbPrimitives.values());

        for (int i = 0; i < primitives.size(); i++) {
            GeometricObject gm = primitives.get(i);
            switch (gm.getType()) {
                case Line:
                    p1 = new DrawPoint(gm.getP1());
                    p2 = new DrawPoint(gm.getP2());
                    loc1 = p1.getLocation();
                    loc2 = p2.getLocation();
                    geometricSoAIndex.addGeometricPoint(p1.id, loc1.x, loc1.y);
                    geometricSoAIndex.addGeometricPoint(p2.id, loc2.x, loc2.y);
                    add(p1);
                    add(p2);
                    ml = new DrawLine(p1, p2);
                    ml.setGeometricId(gm.getPrimitiveId());
                    add(ml);
                    break;
                case Circle:
                    p1 = new DrawPoint(gm.getP1());
                    p2 = new DrawPoint(gm.getP2());
                    add(p1);
                    add(p2);
                    loc1 = p1.getLocation();
                    loc2 = p2.getLocation();
                    geometricSoAIndex.addGeometricPoint(p1.id, loc1.x, loc1.y);
                    geometricSoAIndex.addGeometricPoint(p2.id, loc2.x, loc2.y);
                    mc = new DrawCircle(p1, p2);
                    mc.setGeometricId(gm.getPrimitiveId());
                    add(mc);
                    break;
                case Arc:
                    p1 = new DrawPoint(gm.getP1());
                    p2 = new DrawPoint(gm.getP2());
                    p3 = new DrawPoint(gm.getP3());
                    add(p1);
                    add(p2);
                    add(p3);
                    loc1 = p1.getLocation();
                    loc2 = p2.getLocation();
                    loc3 = p3.getLocation();
                    geometricSoAIndex.addGeometricPoint(p1.id, loc1.x, loc1.y);
                    geometricSoAIndex.addGeometricPoint(p2.id, loc2.x, loc2.y);
                    geometricSoAIndex.addGeometricPoint(p3.id, loc3.x, loc3.y);
                    ma = new DrawArc(p1, p2, p3);
                    ma.setGeometricId(gm.getPrimitiveId());
                    add(ma);
                    break;
                case FreePoint:
                    p1 = new DrawPoint(gm.getP1());
                    add(p1);
                    loc1 = p1.getLocation();
                    geometricSoAIndex.addGeometricPoint(p1.id, loc1.x, loc1.y);
                    fp = new DrawFreePoint(p1);
                    fp.setGeometricId(gm.getPrimitiveId());
                    add(fp);
                    break;
            }
        }

        this.geometricSoAIndex = geometricSoAIndex;
    }

    private void add(DrawLine ml) {
        lineContainer.add(ml);
    }

    private void add(DrawCircle ml) {
        circleContainer.add(ml);
    }

    private void add(DrawFreePoint fp) {
        freePointContainer.add(fp);
    }

    private void add(DrawArc arc) {
        arcContainer.add(arc);
    }

    private void add(DrawPoint p) {
        pointContainer.put(p.id, p);
    }

    void repaintLater() {
        SwingUtilities.invokeLater(this::repaint);
    }

    private Point asPointLocation(int pointId) {
        com.mstruzek.msketch.Point point = ModelRegistry.dbPoint().get(pointId);
        return new Point((int) point.getX(), (int) point.getY());
    }

    @Override
    public void paint(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;
        /// first title and border
        super.paint(g2d);

/// transformation
        AffineTransform tx = (AffineTransform) (G.clone());
        g2d.transform(tx);

        for (DrawLine ml : lineContainer) {
            /// stdout into g2d context
            Point tp2 = ml.p2.getLocation();
            Point tp1 = ml.p1.getLocation();
            g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));

            /// visual guides
            if (controlGuidelines) {
                g2d.setColor(CONTROL_GUIDES_COLOR);

                int pA = ModelRegistry.dbPrimitives.get(ml.getGeometricId()).getA();
                int pB = ModelRegistry.dbPrimitives.get(ml.getGeometricId()).getB();
                //A - p1
                Point tp3 = asPointLocation(pA);
                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp3.x, tp3.y));
                //B -p2
                tp2 = ml.p2.getLocation();
                tp3 = asPointLocation(pB);
                g2d.draw(new Line2D.Double(tp3.x, tp3.y, tp2.x, tp2.y));

                g2d.setColor(PANEL_BORDER_COLOR);
            }
        }

        for (DrawCircle cl : circleContainer) {
            /// stdout into g2d context
            Point tp2 = cl.p2.getLocation();
            Point tp1 = cl.p1.getLocation();
            double r = Math.floor(tp1.distance(tp2));
            g2d.draw(new Ellipse2D.Double(tp1.x - r, tp1.y - r, 2 * r, 2 * r));
            g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));

            if (controlGuidelines) {
                g2d.setColor(CONTROL_GUIDES_COLOR);

                int pA = ModelRegistry.dbPrimitives.get(cl.getGeometricId()).getA();
                int pB = ModelRegistry.dbPrimitives.get(cl.getGeometricId()).getB();
                //A - p1
                Point tp3 = asPointLocation(pA);
                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp3.x, tp3.y));
                //B -p2
                tp2 = cl.p2.getLocation();
                tp3 = asPointLocation(pB);
                g2d.draw(new Line2D.Double(tp3.x, tp3.y, tp2.x, tp2.y));

                g2d.setColor(PANEL_BORDER_COLOR);
            }
        }

        for (DrawArc cl : arcContainer) {
            /// stdout into g2d context
            //System.out.println(cl.p1.getLocation());
            Point tp1 = cl.p1.getLocation();
            Point tp2 = cl.p2.getLocation();
            Point tp3 = cl.p3.getLocation();
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

                int pA = ModelRegistry.dbPrimitives.get(cl.getGeometricId()).getA();
                int pB = ModelRegistry.dbPrimitives.get(cl.getGeometricId()).getB();
                int pC = ModelRegistry.dbPrimitives.get(cl.getGeometricId()).getC();
                int pD = ModelRegistry.dbPrimitives.get(cl.getGeometricId()).getD();
                //A - p1
                tp1 = cl.p1.getLocation();
                tp2 = asPointLocation(pA);
                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));

                //B -p1
                tp1 = cl.p1.getLocation();
                tp2 = asPointLocation(pB);
                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));

                //C - p2
                tp2 = cl.p2.getLocation();
                tp1 = asPointLocation(pC);
                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));
                //D - p3
                tp2 = cl.p3.getLocation();
                tp1 = asPointLocation(pD);
                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));

                g2d.setColor(PANEL_BORDER_COLOR);
            }
        }

        for (DrawFreePoint mfp : freePointContainer) {
            if (controlGuidelines) {
                g2d.setColor(CONTROL_GUIDES_COLOR);

                int pA = ModelRegistry.dbPrimitives.get(mfp.getGeometricId()).getA();
                int pB = ModelRegistry.dbPrimitives.get(mfp.getGeometricId()).getB();
                //A - p1
                Point tp1 = mfp.p1.getLocation();
                Point tp2 = asPointLocation(pA);
                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));
                //B -p1
                tp1 = mfp.p1.getLocation();
                tp2 = asPointLocation(pB);
                g2d.draw(new Line2D.Double(tp1.x, tp1.y, tp2.x, tp2.y));

                g2d.setColor(PANEL_BORDER_COLOR);
            }
        }

        for (DrawPoint point : pointContainer.values()) {
            if (point.hover()) {
                g2d.setColor(Color.DARK_GRAY);
            } else {
                g2d.setColor(Color.BLACK);
            }
            Point tp1 = point.getLocation();
            g2d.fill(new Ellipse2D.Double(tp1.x - pRadius / 2.0, tp1.y - pRadius / 2.0, pRadius, pRadius));
        }

        //Rysujemy osie
        Point p0 = new Point(0, 0);
//        Point px = new Point(0, (int) Math.round(200 / G.getScaleX()));
//        Point py = new Point((int) Math.round(-200 / G.getScaleY()), 0);
        Point px = new Point(0, (int) Math.round(200));
        Point py = new Point((int) Math.round(200), 0);
        g2d.setColor(SYSTEM_COORDINATE_OY_COLOR);
        g2d.draw(new Line2D.Double(p0, px)); //X
        g2d.setColor(SYSTEM_COORDINATE_OX_COLOR);
        g2d.draw(new Line2D.Double(p0, py)); //Y
    }

    private static Point inverseTransform(AffineTransform affineTransform, Point src) {
        Point dest = new Point();
        try {
            affineTransform.createInverse().transform(src, dest);
        } catch (NoninvertibleTransformException e) {
            throw new RuntimeException(e);
        }
        return dest;
    }

    @Override
    public void mouseClicked(MouseEvent e) {
        Point ep = inverseTransform(G, e.getPoint());

        Set<Integer> boundingBox = geometricSoAIndex.findWithInBoundingBox(ep.x, ep.y, pRadius);
        boundingBox.stream().findFirst().ifPresent(pId -> mouseClickedAtPoint(e, pId));

        if (boundingBox.isEmpty() && e.getButton() == MouseEvent.BUTTON3) {
            mpc.clearAll();
            firePropertyChange(Property.KLMN_POINTS, null, mpc.toString());
        }
        repaint();
    }

    private void mouseClickedAtPoint(MouseEvent e, Integer pId) {
        if (e.isAltDown()) {
            mpc.setFreeSlot(pId);
        } else if (e.isControlDown()) {

            GeometricObject parent =
                ModelRegistry.dbPrimitives.values().stream()
                    .filter(p -> Arrays.stream(p.getAllPointsId()).boxed().anyMatch(Predicate.isEqual(pId)))
                    .findFirst()
                    .orElseThrow(() -> new IndexOutOfBoundsException("point id :" + pId));

            // set two consecutive free slots
            switch (parent.getType()) {
                case FreePoint:
                    mpc.setFreeSlot(parent.getP1());
                    break;
                case Arc:
                case Line:
                    if (parent.getP1() == pId) {
                        mpc.setFreeSlot(pId);              // K
                        mpc.setFreeSlot(parent.getP2());   // L
                    } else if (parent.getP2() == pId) {
                        mpc.setFreeSlot(pId);              // K
                        mpc.setFreeSlot(parent.getP1());   // L
                    }
                    break;
                case Circle:
                    mpc.setFreeSlot(parent.getP1()); // M
                    mpc.setFreeSlot(parent.getP2()); // N
                    break;
                case FixLine:
                    // control OXY lines
                    break;
            }
        } else {
            firePropertyChange(Property.SELECTED_POINT, null, ModelRegistry.dbPoint.get(pId).toString());
        }
        firePropertyChange(Property.KLMN_POINTS, null, mpc.toString());
    }

    @Override
    public void mouseEntered(MouseEvent e) {

    }

    @Override
    public void mouseExited(MouseEvent e) {

    }

    @Override
    public void mousePressed(MouseEvent e) {
        if (lastPressed == null) {
            lastPressed = e.getPoint();
        }
    }

    @Override
    public void mouseReleased(MouseEvent e) {
        switch (state) {
            case Normal -> mouseReleasedNormal(e);
            case DrawLine -> mouseReleasedDrawLine(e);
            case DrawCircle -> mouseReleasedDrawCircle(e);
            case DrawArc -> mouseReleasedDrawArc(e);
            case DrawPoint -> mouseReleasedDrawPoint(e);
        }
    }

    private void mouseReleasedDrawPoint(MouseEvent e) {
        if (picked[pId] == null) {
            Point ep = inverseTransform(G, e.getPoint());
            picked[pId++] = new Point(ep);
            /// 3 consecutive picks
            if (pId == 1) {
                Point p1 = picked[0];
                controller.addPoint(new Vector(p1.x, p1.y));
                rebuildViewModel();
                state = ApplicationState.Normal;
                pId = 0;
                picked[0] = null;
                repaint();
            }
        }
    }

    private void mouseReleasedDrawArc(MouseEvent e) {
        if (picked[pId] == null) {
            Point ep = inverseTransform(G, e.getPoint());
            picked[pId++] = new Point(ep);
            /// 3 consecutive picks
            if (pId == 3) {
                Point p1 = picked[0];
                Point p2 = picked[1];
                Point p3 = picked[2];
                controller.addArc(new Vector(p1.x, p1.y), new Vector(p2.x, p2.y), new Vector(p3.x, p3.y));
                rebuildViewModel();
                state = ApplicationState.Normal;
                pId = 0;
                picked[0] = null;
                picked[1] = null;
                picked[2] = null;
                repaint();
            }
        }
    }

    private void mouseReleasedDrawCircle(MouseEvent e) {
        /// 2 consecutive picks
        if (picked[pId] == null) {
            Point ep = inverseTransform(G, e.getPoint());
            picked[pId++] = new Point(ep);
            if (pId == 2) {
                Point p1 = picked[0];
                Point p2 = picked[1];
                controller.addCircle(new Vector(p1.x, p1.y), new Vector(p2.x, p2.y));
                rebuildViewModel();
                state = ApplicationState.Normal;
                pId = 0;
                picked[0] = null;
                picked[1] = null;
                repaint();
            }
        }
    }

    private void mouseReleasedDrawLine(MouseEvent e) {
        /// 2 consecutive picks
        if (picked[pId] == null) {
            Point ep = inverseTransform(G, e.getPoint());
            picked[pId++] = new Point(ep);
            if (pId == 2) {
                Point p1 = picked[0];
                Point p2 = picked[1];
                controller.addLine(new Vector(p1.x, p1.y), new Vector(p2.x, p2.y));
                rebuildViewModel();
                state = ApplicationState.Normal;
                pId = 0;
                picked[0] = null;
                picked[1] = null;
                repaint();
            }
        }
    }

    private void mouseReleasedNormal(MouseEvent e) {
        if (lastPressed != null) {
            lastPressed = null;
        }
        if (pressedId != null) {
            if (e.isPopupTrigger()) {
                popoup.setPointId(pressedId);
                popoup.show(e.getComponent(), e.getX(), e.getY());
            }
            pressedId = null;
        }
    }

    @Override
    public void mouseWheelMoved(MouseWheelEvent e) {
        /// Zoom-In / Zoom-Out at position
        Point point = inverseTransform(G, e.getPoint()); /// new camera point


        double ctrlScale = (e.isControlDown()) ? CTRL_SCALE_SPD2 : CTRL_SCALE_SPD1;

        double len = Math.sqrt(point.x * point.x + point.y * point.y) / (10);

        // scale in the same direction as mouse wheel rotation
        double scale = (SCALE_BASE + ctrlScale * e.getWheelRotation()) / SCALE_BASE;
        int direction = -e.getWheelRotation();
        double tx = direction * point.x * scale * Math.abs(G.getScaleX()) / len;
        double ty = direction * point.y * scale * Math.abs(G.getScaleX()) / len;

        if (Double.isNaN(tx)) tx = jitterSource.nextDouble(JITTER_MAX, JITTER_MIN);
        if (Double.isNaN(ty)) ty = jitterSource.nextDouble(JITTER_MAX, JITTER_MIN);

        /// Zoom-In / Zoom-Out
        G.concatenate(AffineTransform.getTranslateInstance(tx, ty));
        G.concatenate(AffineTransform.getScaleInstance(scale, scale));

        repaint();
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        if(state == ApplicationState.Normal) {

            /// Drag selected point
            if (pressedId == null) {
                Point ep = inverseTransform(G, e.getPoint());
                Set<Integer> withInBoundingBox = geometricSoAIndex.findWithInBoundingBox(ep.getX(), ep.getY(), pRadius);
                if (!withInBoundingBox.isEmpty()) {
                    withInBoundingBox.stream().findFirst().ifPresent(pointId -> pressedId = pointId);
                }
            }

            /// Update location of selected point
            if (pressedId != null) {
                Point ep = inverseTransform(G, e.getPoint());
                Point point = pointContainer.get(pressedId);
                point.setLocation(ep);
                updateGeometricSOAIndex();
                repaint();
                return;
            }

            /// Move coordinate view  ( `pan ),  Cx, Cy ,
            if (lastPressed != null ) {
                Point sp1 = e.getPoint();
                Point dvec = new Point(sp1.x - lastPressed.x, sp1.y - lastPressed.y);
                // R0.1 * R1.2 * R2.3 ....
                G.concatenate(AffineTransform.getTranslateInstance(dvec.getX() / G.getScaleX(), dvec.getY() / G.getScaleY()));
                lastPressed = e.getPoint();
                repaint();
            }
        }
    }


    @Override
    public void mouseMoved(MouseEvent e) {
        Point ep = inverseTransform(G, e.getPoint());
        firePropertyChange(Property.CURRENT_POSITION, null, String.format("X : %1$5.3f , Y : %2$5.3f", ep.getX(), ep.getY()));

        // Hover Reaction at point position
        Optional<Integer> hoverPoint = geometricSoAIndex.findWithInBoundingBox(ep.x, ep.y, pRadius).stream().findFirst();
        hoverPoint.ifPresent(pointId -> mouseHoverAtPoint(e, pointId));

        if (hoverState && (currentTimeMillis() - selectionTime) > AWAIT_HOVER_TIMEOUT) {
            pointContainer.values().forEach(point -> point.setHover(false));
            hoverState = false;
            repaint();
        }
    }

    private void mouseHoverAtPoint(MouseEvent e, int pointId) {
        pointContainer.get(pointId).setHover(true);
        //pointContainer.get(pointId).setHover(true);
        hoverState = true;
        selectionTime = currentTimeMillis();
        repaint();
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
        firePropertyChange(KLMN_POINTS, null, mpc.toString());
    }

}

