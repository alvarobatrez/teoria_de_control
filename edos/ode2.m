function [t_out, x_out] = ode2(fun, t0, h, tf, x0)

x0 = x0(:);
N = floor((tf - t0) / h) + 1;
x_out = zeros(N, length(x0));
t_out = (t0:h:tf)';

t = t0;
x = x0;
x_out(1,:) = x';

for i = 2 : N
    s1 = fun(t,x);
    s2 = fun(t+h/2, x+h/2*s1);
    x = x + h*s2;
    x_out(i,:) = x';
    t = t + h;
end