package pl.struzek.graphic;

public class MyFreePoint extends MyDrawElement{

	MyPoint p1;
	
	public MyFreePoint(int x1,int y1){
		p1 = new MyPoint(x1,y1);
	}
	public MyFreePoint(MyPoint tp1){
		p1=tp1;

	}
	public MyPoint[] getPoints(){
		MyPoint[] mp =  {p1};
		return mp;
	}
}