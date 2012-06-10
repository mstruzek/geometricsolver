package pl.struzek.msketch;

import java.awt.Container;
import java.awt.Dimension;

import javax.swing.JFrame;
import javax.swing.JPanel;

public class Rysunek extends JFrame {

	/**
	 * FIXME KOniecznie trzeba dodac relaksacje wiezow
	 */
	private static final long serialVersionUID = 1L;
	Container cp = getContentPane();

	Rysunek(){
		super("MSketch");
		//cp.setLayout(null);
		//jakby nasz szkicownik
		final JPanel panel = new JPanel();
		panel.setLayout(null);
		Sketch2D sk = new Sketch2D(600,600);
		panel.add(sk);
		cp.add(panel);

		
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		//setSize(600, 600);
		setPreferredSize(new Dimension(600, 600));
		pack();
		setVisible(true);
		
	}
	

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		

		new Rysunek();
		
	}

}
