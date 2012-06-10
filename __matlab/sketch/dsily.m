function dQ = dsily(w)
%% zwracam macierz d(Q)/dq

%wczytaj dane
dane;

dQ = [-K K o o ,
       K -2*K K o ,
       o K -2*K K ,
       o o  K K ];