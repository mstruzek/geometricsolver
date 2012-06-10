package pl.struzek.graphic;

public class MyCircle extends MyDrawElement{
	
	MyPoint p1,p2;
	
	public MyCircle(MyPoint tp1,MyPoint tp2){
		p1=tp1;
		p2=tp2;
	}
	public MyPoint[] getPoints(){
		MyPoint[] mp =  {p1,p2};
		return mp;
	}
}
