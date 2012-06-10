function Q = sily(w)
%% funkcja zwraca wektor sil w zaleznosci od wektora w

%wczytujemy dane
dane;

%nadanie wektorowi 'w' czytelnych zmiennych
q1 = w(1:2); q2 = w(3:4); q3 = w(5:6); q4 = w(7:8);
%bez mnoznikow Lagrange'a

% odleglosci
d12 = sqrt((q2-q1)'*(q2-q1));
d23 = sqrt((q3-q2)'*(q3-q2));
d34 = sqrt((q4-q3)'*(q4-q3));

Q=zeros(8,1);

%SILY
Q(1:2) = (q2-q1)*k*(1-d0/d12); % Q1
Q(3:4) = -(q2-q1)*k*(d12-d0)/d12 + (q3-q2)*k*(d23-d0)/d23; %Q2
Q(5:6) = -(q3-q2)*k*(d23-d0)/d23 + (q4-q3)*k*(d34-d0)/d34; %Q3
Q(7:8) = - (q4-q3)*k*(d34-d0)/d34; %Q4

%czyszczenie z nieskonczonosci np dzielenie przez zero
Q(isnan(Q)) = 0;


