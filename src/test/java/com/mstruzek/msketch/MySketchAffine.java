package com.mstruzek.msketch;

import com.mstruzek.graphic.GeometricSOAIndex;

import javax.swing.*;
import javax.swing.event.MouseInputListener;
import java.awt.Point;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.geom.AffineTransform;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.geom.NoninvertibleTransformException;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;
import java.util.logging.Logger;


public class MySketchAffine extends JPanel implements MouseInputListener, MouseWheelListener {
    public static final int S_WIDTH = 1320;
    public static final int S_HEIGHT = 1000;
    public static final Color SYSTEM_COORDINATE_OX_COLOR = Color.RED;
    public static final Color SYSTEM_COORDINATE_OY_COLOR = Color.GREEN;
    public static final Color PANEL_BORDER_COLOR = Color.DARK_GRAY;
    public static final Color SKETCH_PANE_BACKGROUND_COLOR = new Color(160, 192, 222);
    public static final Logger WHEEL = Logger.getLogger("::");
    public static final int RANDOM_POINTS = 360;
    public static final int BOUNDS = 200;


    /// Zoom-In / Zoom-Out with left Ctrl button
    public static final double CTRL_SCALE_FAST = 24.0;
    public static final double CTRL_SCALE_SLOW = 8.0;
    public static final double SCALE_BASE = 100.0;

    /**
     * aktualny stan podczas klikania myszka
     */

    /// plane at distance z = 100 [units]
    private volatile AffineTransform G = new AffineTransform();

    public MySketchAffine() {
        super();
        setLayout(null);
        setPreferredSize(new Dimension(S_WIDTH, S_HEIGHT));
        setBackground(SKETCH_PANE_BACKGROUND_COLOR);

        G.scale(1.0, -1.0);
        G.translate(30.0, -800.0);

        setBorder(BorderFactory.createTitledBorder(BorderFactory.createLineBorder(PANEL_BORDER_COLOR), "Szkicownik"));
        addMouseListener(this);
        addMouseMotionListener(this);
        addMouseWheelListener(this);

        refreshModel();
    }

    void repaintLater() {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                MySketchAffine.this.repaint();
            }
        });
    }

    private final int r = 4;
    private HashMap<Integer, java.awt.Point> model = new HashMap<>();

    private volatile GeometricSOAIndex geometricSoAIndex = null;

    private final Color[] colors = new Color[]{ Color.BLUE, Color.GREEN, Color.BLACK, Color.GRAY};


    /// rebuild model and geometric r-tree
    private void refreshModel() {
        GeometricSOAIndex geometricSoAIndex = new GeometricSOAIndex();
        for (int i = 0; i < RANDOM_POINTS; i++) {
            int x = new Random().nextInt(-BOUNDS, BOUNDS);
            int y = new Random().nextInt(-BOUNDS, BOUNDS);
            java.awt.Point point = new java.awt.Point(x, y);
            model.put(i, point);
            geometricSoAIndex.addGeometricPoint(i, x, y);
        }
        this.geometricSoAIndex = geometricSoAIndex;
    }

    /// Internally used after location changed
    private void updateGeometricTree() {
        GeometricSOAIndex geometricSoAIndex = new GeometricSOAIndex();
        for (int i = 0; i < RANDOM_POINTS; i++) {
            Point p = model.get(i);
            geometricSoAIndex.addGeometricPoint(i, p.x, p.y);
        }
        this.geometricSoAIndex = geometricSoAIndex;
    }

    @Override
    public void paint(Graphics g) {
        //promien punktu
        Graphics2D g2d = (Graphics2D) g;

        AffineTransform tx = (AffineTransform) (G.clone());
        super.paint(g2d);
        g2d.transform(tx);

        for (int i = 0; i < RANDOM_POINTS; i++) {
            Color color = colors[i % colors.length];
            g2d.setColor(color);
            Point point = model.get(i);
            g2d.fill(new Ellipse2D.Double(point.x - r / 2, point.y - r / 2, r, r));
        }

        /// Rysujemy osie
        java.awt.Point p0 = new java.awt.Point(0, 0);
        java.awt.Point px = new java.awt.Point(0, (int) Math.round(200 / tx.getScaleX()));
        java.awt.Point py = new java.awt.Point((int) Math.round(-200 / tx.getScaleY()), 0);

        g2d.setColor(SYSTEM_COORDINATE_OY_COLOR);
        g2d.draw(new Line2D.Double(p0, px)); //X
        g2d.setColor(SYSTEM_COORDINATE_OX_COLOR);
        g2d.draw(new Line2D.Double(p0, py)); //Y

        /// g2d.getTransform(); // ostatnia KOPIA trnsformacji
        /// g2d.setTransform(g2d.getTransform()); // UPDATE transformation
    }


    // structure use for transformation of coordinate system
    private volatile Point lastPressed = null;

    /// hold last point id selected for drag operation
    private volatile Integer pressedId = null;

    private void inverseTransform(AffineTransform affineTransform, Point src, Point dest) {
        try {
            affineTransform.createInverse().transform(src, dest);
        } catch (NoninvertibleTransformException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void mouseClicked(MouseEvent e) {
        if(e.isControlDown()) {
            Point pS = e.getPoint();
            inverseTransform(G, pS, pS);
            Set<Integer> withInBoundingBox = geometricSoAIndex.findWithInBoundingBox(pS.getX(), pS.getY(), r);
            //System.out.println("add mpc point" + pS + withInBoundingBox);
        }
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
        if (lastPressed != null) {
            lastPressed = null;
        }

        if (pressedId != null) {
            pressedId = null;
        }
    }

    @Override
    public void mouseDragged(MouseEvent e) {

        /// Drag selected point
        if(pressedId == null) {
            Point pS = e.getPoint();
            inverseTransform(G, pS, pS);
            Set<Integer> withInBoundingBox = geometricSoAIndex.findWithInBoundingBox(pS.getX(), pS.getY(), r);
            if(!withInBoundingBox.isEmpty()) {
                withInBoundingBox.stream().findFirst().ifPresent(pointId -> pressedId = pointId);
            }
        }

        /// Update location of selected point
        if (pressedId != null) {
            Point ep = e.getPoint();
            inverseTransform(G, ep, ep);
            Point point = model.get(pressedId);
            point.setLocation(ep);
            updateGeometricTree();
            repaint();
            return;
        }

        /// Move coordinate view  ( `pan ),  Cx, Cy ,
        if (lastPressed != null) {
            Point sp1 = e.getPoint();
            Point dvec = new Point(sp1.x - lastPressed.x, sp1.y - lastPressed.y);
            // R0.1 * R1.2 * R2.3 ....
            G.concatenate(AffineTransform.getTranslateInstance(dvec.getX() / G.getScaleX(), dvec.getY() / G.getScaleY()));
            lastPressed = e.getPoint();
            repaint();
        }
    }

    @Override
    public void mouseMoved(MouseEvent e) {
    }


    static class PointUtils {

        static public Point newPoint(double x , double y) {
            return new Point((int) x, (int) y);
        }

        static public Point scale(Point p , double scale) {
            return new Point((int) (p.x* scale), (int) (p.y * scale));
        }

        static public double length(Point point) {
            double product = product(point, point);
            return Math.sqrt(product);
        }

        static public double product(Point p1, Point p2) {
            return p1.x * p1.x + p2.y * p2.y;
        }



    }

    @Override
    public void mouseWheelMoved(MouseWheelEvent e) {

        // +1.0 -1.0
        int direction = e.getWheelRotation();

        /// Zoom-In / Zoom-Out at position
        Point point = e.getPoint(); /// new camera point
        inverseTransform(G, point, point);

        double ctrlScale = (e.isControlDown()) ? CTRL_SCALE_FAST : CTRL_SCALE_SLOW;

        // scale in the same direction as mouse wheel rotation
        double scale = (SCALE_BASE + ctrlScale * e.getWheelRotation()) / SCALE_BASE;

        // towards center point cx, cy of sketch panel
        int x = (int) (Math.signum(G.getScaleX()) * getWidth() / 2 - e.getPoint().x);
        int y = (int) (Math.signum(G.getScaleY()) * (getHeight() / 2 - e.getPoint().y));
        var vector = new Point(x,y);
        double length = PointUtils.length(vector);

        double factor = length * 0.00005 * ctrlScale;
        var translateToSr = PointUtils.scale(vector, factor * direction );

        /// Zoom-In / Zoom-Out
        G.concatenate(AffineTransform.getTranslateInstance(translateToSr.x, translateToSr.y));
        G.concatenate(AffineTransform.getScaleInstance(scale, scale));

        repaint();
    }


    public static void main(String[] args) {

        JFrame frame = new JFrame("window");
        Container contentPane = frame.getContentPane();
        JPanel panel = new JPanel();
        new BoxLayout(panel, BoxLayout.X_AXIS);
        panel.add(new MySketchAffine());
        contentPane.add(panel);
        frame.pack();
        frame.setVisible(true);
    }


    public static void mainW(String[] args) throws NoninvertibleTransformException {
        var R10 = new AffineTransform();
        R10.translate(30.0, -600.0);
        /// second
        double Yscale = -1.0;
        R10.scale(1.0, Yscale);

        var p1 = new Point(30, 30);
        var p1_0 = new Point(0, 0);
        R10.transform(p1, p1_0);
        var p1_1 = new Point(0, 0);
        R10.transform(p1_0, p1_1);

        R10.translate(BOUNDS, Yscale * BOUNDS); // tranzlacja w ukladzie 0 (scale)
//        R10.invert();
//        R10.translate(100,100); // tranzlacja w ukladzie 1 (scale)
//        R10.invert();

        //R10.invert();
        R10.scale(.1, .1);
        //R10.invert();
        R10.transform(p1, p1_0);
        // v0 = R0.1 * v1

        // v0 = R0.1 * R1.2 * v2
        return;

    }

    public static void main__base(String[] args) throws NoninvertibleTransformException {

        var R01 = new AffineTransform();

        R01.translate(60, 10);

        var R12 = new AffineTransform();
        R12.translate(30, 10); // 90,20

        var p2 = new Point(30, 30);
        var p2_1 = new Point(0, 0);
        var p2_0 = new Point(0, 0);

        // 120, 50
        R12.transform(p2, p2_1);
        R01.transform(p2_1, p2_0);
        Logger.getLogger("").warning(p2_0.toString());

        var R02 = new AffineTransform();
        R02.concatenate(R01);
        R02.concatenate(R12);

        p2_0 = new Point(0, 0);
        R02.transform(p2, p2_0);
        Logger.getLogger("").warning(p2_0.toString());

        // polozenie ukladu wspolrzednych `2 w ukladzie `0.
        var tX = R02.getTranslateX();
        var tY = R02.getTranslateX();

        var R20 = R02.createInverse();

        return;

    }

}

