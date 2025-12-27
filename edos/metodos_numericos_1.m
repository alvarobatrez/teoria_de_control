close all; clear, clc

t0 = 0;
tf = 0.5;
h = 0.1;
y0 = 1;

fun = @(t,y) 2*y;

%% Euler
[~,y1] = ode1(fun, t0, h, tf, y0);

%% Punto Medio
[~,y2] = ode2(fun, t0, h, tf, y0);

%% Runge-Kutta 4
[t,y3] = ode4(fun, t0, h, tf, y0);

%% Solucion analitica
y = exp(2*t);

figure, hold on, grid on
plot(t,y,'--r')
plot(t,y1,'g')
plot(t,y2,'b')
plot(t,y3,'y')

legend('Solución analítica', 'Euler', 'Punto medio', 'RK4', 'Location','northwest')