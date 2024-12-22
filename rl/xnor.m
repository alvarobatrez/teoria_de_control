close all; clear, clc

eta = 0.1;
epochs = 30000;

X = [1 0 0; 1 0 1; 1 1 0; 1 1 1];
Y = [1; 0; 0; 1];

w1 = [0.3 -0.1 0.2; 0.1 0.2 0];
w2 = [-0.3; 0.1; -0.2];

sig = @(x) 1 ./ (1 + exp(-x));
dsig = @(x) exp(-x) ./ (1 + exp(-x)).^2;

error = zeros(epochs, 1);

for epoch = 1 : epochs
    e = 0;
    for i = 1 : length(X)
        x = X(i,:)';
        y = Y(i);

        h = [1; sig(w1 * x)];
        o = sig(w2' * h);

        e = e + 0.5 * (y - o).^2;

        delta_o = (o - y) * dsig(w2' * h);
        delta_h = dsig(w1 * x) .* (w2(2:end) * delta_o);

        w2 = w2 - eta * delta_o * h;
        w1 = w1 - eta * delta_h * x';
    end

    error(epoch) = e;
end

disp('Pesos capa oculta:')
disp(w1)
disp('Pesos capa salida:')
disp(w2)

disp('Resultados:')
for i = 1 : length(X)
    x = X(i,:);
    y = Y(i);

    h = [1; sig(w1 * x')];
    o = sig(w2' * h);

    fprintf('Entrada: [%d %d], Salida: %.1f\n', X(i,1), X(i,2), o)
end

plot(1:epochs, error)
title('Función Costo'), xlabel('Epocas'), ylabel('Error'), grid on