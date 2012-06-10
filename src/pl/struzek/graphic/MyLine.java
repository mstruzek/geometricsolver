package pl.struzek.graphic;

public class MyLine extends MyDrawElement{

	MyPoint p1,p2;
	
	public MyLine(int x1,int y1,int x2,int y2){
		p1 = new MyPoint(x1,y1);
		p2 = new MyPoint(x2,y2);
	}
	public MyLine(MyPoint tp1,MyPoint tp2){
		p1=tp1;
		p2=tp2;
	}
	public MyPoint[] getPoints(){
		MyPoint[] mp =  {p1,p2};
		return mp;
	}

}
