package pl.struzek.graphic;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;

import javax.swing.BorderFactory;
import javax.swing.JFrame;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.event.MouseInputListener;
import javax.swing.table.TableColumn;

import pl.struzek.controller.Controller;
import pl.struzek.msketch.Model.MyTableModel;

public class MyTables extends JPanel implements MouseInputListener {

    /**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/** constraint */
	MyTableModel mtm = null; 
	/** primitives */
	MyTableModel ptm = null; 
	/** parameters */
	MyTableModel vtm = null; 
    
    JTable table = null;
    JTable table2 = null;
    JTable table3 = null;
    
    MPopup popoupConstaint ;
    MPopup popupPrimitives ;
   // MPopup popupParameters = new MPopup(vtm,table3);
    
    public MyTables(MyTableModel constraint,MyTableModel primitives,MyTableModel parameters) {
        super();
        setPreferredSize(new Dimension(420, 470));
        //(new BorderLayout());
        this.mtm = constraint;
        this.ptm = primitives;
        this.vtm = parameters;
        
        table = new JTable(mtm);
        table2 = new JTable(ptm);
        table3 = new JTable(vtm);
        
		table.setPreferredScrollableViewportSize(new Dimension(400, 150));
		table2.setPreferredScrollableViewportSize(new Dimension(400, 150));
		table3.setPreferredScrollableViewportSize(new Dimension(400, 120));
		
        //table.setFocusable(false);
        //table2.setFocusable(false);
        //table3.setFocusable(false);
        
        table.addMouseListener(this);
        table.addMouseMotionListener(this);

        table2.addMouseListener(this);
        table2.addMouseMotionListener(this);  
        
        table3.addMouseListener(this);
        table3.addMouseMotionListener(this);        
        
        TableColumn column = null;
        for (int i = 0; i < 1; i++) { //8
            column = table.getColumnModel().getColumn(i);
            switch(i){
            case 0:
            	column.setPreferredWidth(8);
            	break;
            case 1:
            	column.setPreferredWidth(120);
            	break;
            case 2:
            	column.setPreferredWidth(8);
            	break;
            case 3:
            	column.setPreferredWidth(8);
            	break;
            case 4:
            	column.setPreferredWidth(8);
            	break;           	
            case 5:
            	column.setPreferredWidth(8);
            	break;
            case 6:
            	column.setPreferredWidth(8);
            	break;   
            case 7:
            	column.setPreferredWidth(60);
            	break; 
            }
        }
        TableColumn column2 = null;
        for (int i = 0; i < 2; i++) {
            column2 = table2.getColumnModel().getColumn(i);
            switch(i){
            case 0:
            	column2.setPreferredWidth(10);
            	break;
            case 1:
            	column2.setPreferredWidth(200);
            	break;
            case 2:
            	column2.setPreferredWidth(10);
            	break;
            case 3:
            	column2.setPreferredWidth(10);
            	break;
            case 4:
            	column2.setPreferredWidth(10);
            	break;           	   
            }
        }     
        JScrollPane tablePane = new JScrollPane(table);
        JScrollPane tablePane2 = new JScrollPane(table2);
        JScrollPane tablePane3 = new JScrollPane(table3);
        

        //Do the layout.

        add(tablePane);
        add(tablePane2);
        add(tablePane3);
        
        popoupConstaint = new MPopup(mtm,table);
        popupPrimitives = new MPopup(ptm,table2);

    }


    @Override
	public void mouseReleased(MouseEvent e) {
    	//System.out.println(e.getSource());
    	if(((JTable)e.getSource()).equals(table)){
			popoupConstaint.rebuild();
			popoupConstaint.show(e.getComponent(),e.getX(),e.getY());   		
    	}else if(((JTable)e.getSource()).equals(table2)){
    		popupPrimitives.rebuild();
    		popupPrimitives.show(e.getComponent(),e.getX(),e.getY());   
    	}else if(((JTable)e.getSource()).equals(table3)){
    		//popupParameters.rebuild();
    		//popupParameters.show(e.getComponent(),e.getX(),e.getY());   
    	}
    	
	}

	@Override
	public void mouseClicked(MouseEvent arg0) {}

	@Override
	public void mouseEntered(MouseEvent arg0) {}

	@Override
	public void mouseExited(MouseEvent arg0) {}

	@Override
	public void mousePressed(MouseEvent arg0) {}

	@Override
	public void mouseDragged(MouseEvent arg0) {	}
	@Override
	public void mouseMoved(MouseEvent arg0) {}
	
	/**
	 * Klasa wenwetrzna odpowiedzialna za 
	 * widok podczas usuwania elementu z tabeli
	 * wystarczy ze tabela implementuje interfejs 
	 * {@link TableModelRemovable}
	 * @author root
	 *
	 */
	class MPopup extends JPopupMenu implements ActionListener{
		
		TableModelRemovable model;
		JTable tab;
		
		MPopup(TableModelRemovable m,JTable table){
			super();
			this.model = m;
			this.tab = table;
		}
		
		public void rebuild(){
			this.removeAll();
			JMenuItem bt = new JMenuItem("DELETE");
			bt.addActionListener(this);
			this.add(bt);
		}
	
		@Override
		public void actionPerformed(ActionEvent e) {
			model.remove(tab.getSelectionModel().getMinSelectionIndex());
		}
	}
	public static void main(String[] args) {
		
		JFrame frm = new JFrame("test");
		
		Controller controller =  new Controller();
		
		MyTables mt = new MyTables(controller.getConstraintTableModel(),controller.getPrimitivesTableModel(),controller.getParametersTableModel());
		
		frm.getContentPane().add(new JScrollPane(mt));
		
		frm.pack();
		frm.setVisible(true);
		
	}
}
