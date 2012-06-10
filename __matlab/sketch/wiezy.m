function W = wiezy(w)
%% Zwracan wiezy wartosc wiezow od w =[q,a] -wektor kolumnowy
%  q- wspolrzedne uogulnione , a- mnozniki lagrange'a

%wczytujemy dane
dane;

%nadanie wektorowi 'w' czytelnych zmiennych
q1 = w(1:2); q2 = w(3:4); q3 = w(5:6); q4 = w(7:8);
a1 = w(9); a2 = w(10); a3 = w(11); a4 = w(12); a5 = w(13); a6 = w(14); a7 = w(15);

W = zeros(7,1);
%wiezy
W(1:2) = q1;
W(3:4) = q2-s;
W(5) = (q3-q2)'*Yu; % prostopadlosc do osi Y
W(6:7) = q4-r; % zaczepienie
