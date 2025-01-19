close all; clear, clc

X = [0 0; 0 1; 1 0; 1 1];
Y = [1; 0; 0; 1];

[~, num_inputs] = size(X);

eta = 0.001;
optimizer = 'adam';
loss_function = 'mse';
epochs = 1000;

layers = {{10, 'relu'} {10, 'relu'} {1, 'sigmoid'}};

model = NeuralNetwork(num_inputs, layers);
model = model.compile(eta, optimizer, loss_function);
[model, history] = model.train(X, Y, epochs);
y_pred = model.predict(X);

disp('Resultados')
disp(y_pred)

plot(1:epochs, history), title('Función Costo')
xlabel('Épocas'), ylabel('Error (MSE)')