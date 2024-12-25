close all; clear, clc

X = [0 0; 0 1; 1 0; 1 1];
Y = [1; 0; 0; 1];

epochs = 30000;

layers = [
    featureInputLayer(2)
    fullyConnectedLayer(2); sigmoidLayer
    fullyConnectedLayer(1); sigmoidLayer
    ];

options = trainingOptions('sgdm', 'MaxEpochs', epochs, 'Plots', 'training-progress');

net = trainnet(X, Y, layers, "mse", options);

disp('Resultados:')
for i = 1 : length(X)
    x = X(i,:);
    o = predict(net,x);

    fprintf('Entrada: [%d %d], Salida: %.1f\n', X(i,1), X(i,2), o)
end