close all; clear, clc

[x, y] = meshgrid(-4:0.5:4, -4:0.5:4);

s = y.^2 - x.^2;

L = sqrt(1 + s.^2); % L es para reescalar la magnitud de las flechas

quiver(x,y, 1./L, s./L, 0.5) % 0.5 se refiere al tamaño de las flechas
xlabel('x'), ylabel('y'), title('Campo de dirección')