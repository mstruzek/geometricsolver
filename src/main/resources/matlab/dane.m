%% Dane do programu

k=1; % N/m
d0=1 ; % m
s=[d0 d0]'; % wektor wspolrzednej q1
r=[3*d0 0]';% wektor wspolrzednej q4
Xu = [1 0]';% wersor osi X
Yu = [0 1]';% wersor osi Y

K = k*eye(2,2); % macierz pomocnicza
I = eye(2,2);
o = zeros(2,2);