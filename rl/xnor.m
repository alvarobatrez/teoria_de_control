close all; clear, clc

X = [0 0; 0 1; 1 0; 1 1];
Y = [1; 0; 0; 1];

[m, n] = size(X);

learning_rate = 0.1;
epochs = 1e4;

layers = [16 4 1]; % 2 capas ocultas y 1 capa de salida

w1 = randn(layers(1), n+1);
w2 = randn(layers(2), layers(1)+1);
w3 = randn(layers(3), layers(2)+1);

total_loss = zeros(epochs, 1);

X = [ones(m, 1) X]; % sesgo

for epoch = 1 : epochs
    loss = 0;
    for i = 1 : m
        x = X(i, :)';
        y = Y(i, :)';

        % Propagacion hacia adelante
        z1 = w1 * x;
        a1 = [1; sigmoid(z1)];
        z2 = w2 * a1;
        a2 = [1; sigmoid(z2)];
        z3 = w3 * a2;
        y_pred = sigmoid(z3);

        % Error cuadratico medio MSE
        loss = loss + sum(0.5 * (y_pred - y).^2);

        % Retropropagacion (gradiente descendente)
        delta3 = sigmoid_derivative(y_pred) .* (y_pred - y);
        delta2 = sigmoid_derivative(a2(2:end)) .* (w3(:,2:end)' * delta3);
        delta1 = sigmoid_derivative(a1(2:end)) .* (w2(:,2:end)' * delta2);

        % Actualización de los pesos
        w1 = w1 - learning_rate * delta1 * x';
        w2 = w2 - learning_rate * delta2 * a1';
        w3 = w3 - learning_rate * delta3 * a2';
    end

    total_loss(epoch) = loss / m;
end

disp('Resultados:')
for i = 1 : m
    x = X(i, :)';

    z1 = w1 * x;
    a1 = [1; sigmoid(z1)];
    z2 = w2 * a1;
    a2 = [1; sigmoid(z2)];
    z3 = w3 * a2;
    y_pred = sigmoid(z3)';
    fprintf('Entrada: [%d %d], Salida: [%.1f]\n', X(i,2:end), y_pred)
end

plot(1:epochs, total_loss), grid on
title('Función Costo'), xlabel('Epocas'), ylabel('Error MSE')