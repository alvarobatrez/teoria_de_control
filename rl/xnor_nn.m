close all; clear, clc

X = [0 0; 0 1; 1 0; 1 1];
Y = [1; 0; 0; 1];

layers = {{16, 'sigmoid'} {4, 'sigmoid'} {1, 'sigmoid'}};
epochs = 1e4;
optimizer = 'sgd';
loss_function = 'mse';

[m, num_inputs] = size(X);

model = NeuralNetwork(num_inputs, layers);
model = model.compile(optimizer, loss_function);
[model, history] = model.train(X, Y, epochs);

disp('Resultados:')
for i = 1 : m
    y_pred = model.predict(X(i,:));
    fprintf('Entrada: [%d %d], Salida: [%.2f]\n', X(i,:), y_pred)
end

plot(1:epochs, history), grid on
title('Función Costo'), xlabel('Epocas'), ylabel('Error (MSE)')