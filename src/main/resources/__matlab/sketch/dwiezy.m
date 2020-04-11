function dW = dwiezy(w)
%% Macierz Jakobiego dla wiezow

%wczytujemy dane
dane;
%nadanie wektorowi 'w' czytelnych zmiennych
q1 = w(1:2); q2 = w(3:4); q3 = w(5:6); q4 = w(7:8);
a1 = w(9); a2 = w(10); a3 = w(11); a4 = w(12); a5 = w(13); a6 = w(14); a7 = w(15);

dW = zeros(7,8);

dW(1:2,:) = [I o o o];
dW(3:4,:) = [o I o o];
dW(5,:)   = [0 0 -Yu' Yu' 0 0];
dW(6:7,:) = [o o o I];

