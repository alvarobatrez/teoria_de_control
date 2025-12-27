function [t_out, x_out] = ode1(fun, t0, h, tf, x0)

x0 = x0(:);
N = floor((tf - t0) / h) + 1;
x_out = zeros(N, length(x0));
t_out = (t0:h:tf)';

t = t0;
x = x0;
x_out(1,:) = x';

for i = 2 : N
    s = fun(t,x);
    x = x + h*s;
    x_out(i,:) = x';
    t = t + h;
end