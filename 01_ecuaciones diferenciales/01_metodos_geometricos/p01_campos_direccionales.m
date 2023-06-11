close all; clear, clc

[x, y] = meshgrid(-2:0.2:2, -2:0.2:2);

s = 3*x - y;

L = sqrt(1 + s.^2); % L es para reescalar la magnitud de las flechas

quiver(x,y, 1./L, s./L, 0.5) % 0.5 se refiere al tamaño de las flechas

axis([-2.2 2.2 -2.2 2.2])
xlabel('x'), ylabel('y'), title('Campo de dirección')