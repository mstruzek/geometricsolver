function [w,tW,tdW] = Newton(w0)
%% funkcja do rozwiazywnaia zadania sketch 
% w0 - poczatkowe
 
w=w0;

%% A*dw=B
% macierz stala - ale tylko dla tego przypadku
A = [ dsily(w) dwiezy(w)';
      dwiezy(w) zeros(7,7)];
  
for i=1:30
   
    %mnozniki lagrange
    a = w0(9:15);
    
    B = [ -dwiezy(w)'*a-sily(w);
            -wiezy(w)]; %
       
       %rozwiazujemy zadanie Newtona
       dw=inv(A)*B;
       w=w+dw;
       tW(i,:)=w;
       tdW(i,:)=dw;
    
       
       %Warunek wyjscia z petli czesc od zmiennych
       dq=dw(1:8);
       eps=sqrt(dq'*dq);
       if eps<10e-6
           break;
       end
end