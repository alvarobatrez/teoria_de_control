close all; clear, clc

eta = 0.1;
epochs = 30000;

X = [0 0; 0 1; 1 0; 1 1];
Y = [1; 0; 0; 1];

[m, ~] = size(X);

w1 = rand(2, 3)*2-1;
w2 = rand(1, 3)*2-1;

sig = @(x) 1 ./ (1 + exp(-x));
dsig = @(x) exp(-x) ./ (1 + exp(-x)).^2;

error = zeros(epochs, 1);

X = [ones(m, 1) X];

for epoch = 1 : epochs
    e = 0;
    for i = 1 : m
        x = X(i,:)';
        y = Y(i);
        
        % Propagacion hacia adelante
        z1 = w1 * x;
        h = [1; sig(z1)];
        z2 = w2 * h;
        o = sig(z2);
    
        % Calculo del error
        e = e + 0.5 * (y - o).^2;

        % Retropropagacion
        delta_o = (o - y) * dsig(z2);
        delta_h = dsig(z1) .* (w2(:,2:end)' * delta_o);

        % Actualización de los pesos
        w2 = w2 - eta * delta_o * h';
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
    x = X(i,:)';
    y = Y(i);

    z1 = w1 * x;
    h = [1; sig(z1)];
    z2 = w2 * h;
    o = sig(z2);

    fprintf('Entrada: [%d %d], Salida: %.1f\n', X(i,1), X(i,2), o)
end

plot(1:epochs, error)
title('Función Costo'), xlabel('Epocas'), ylabel('Error'), grid on