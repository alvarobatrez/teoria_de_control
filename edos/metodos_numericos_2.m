close all; clear, clc

function dy = fun(~, y)
    dy = [y(2)
          (1-y(1).^2)*y(2)-y(1)];
end

tspan = [0 20];
y0 = [2; 0];

[t, y] = ode45(@fun, tspan, y0);

plot(t,y), grid on
legend('y0','y1')