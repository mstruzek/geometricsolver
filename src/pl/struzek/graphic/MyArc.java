package pl.struzek.graphic;

public class MyArc extends MyDrawElement{

	MyPoint p1,p2,p3;
	
	public MyArc(int x1,int y1,int x2,int y2,int x3,int y3){
		p1 = new MyPoint(x1,y1);
		p2 = new MyPoint(x2,y2);
		p2 = new MyPoint(x3,y3);
	}
	public MyArc(MyPoint tp1,MyPoint tp2,MyPoint tp3){
		p1=tp1;
		p2=tp2;
		p3=tp3;
	}
	public MyPoint[] getPoints(){
		MyPoint[] mp =  {p1,p2,p3};
		return mp;
	}
}