classdef NeuralNetwork
    properties
        layers = {}
        learning_rate
        optimizer
        loss_function
        num_layers
    end

    methods
        function model = NeuralNetwork(input_size, layers)
            model.num_layers = length(layers);
            for i = 1 : model.num_layers
                if i == 1
                    weights = randn(layers{i}{1}, input_size+1) * sqrt(2 / input_size);
                else
                    weights = randn(layers{i}{1}, layers{i-1}{1}+1) * sqrt(2 / layers{i-1}{1});
                end
                model.layers{i} = struct('weights', weights, 'activation', layers{i}{2});
            end
        end

        function model = compile(model, optimizer, loss_function)
            model.optimizer = optimizer;
            model.loss_function = loss_function;

            if strcmp(optimizer, 'sgd')
                model.learning_rate = 0.1;
            end
        end

        function outputs = forward(model, x)
            outputs = {x};
            
            for i = 1 : model.num_layers
                z = model.layers{i}.weights * outputs{i};
                
                if strcmp(model.layers{i}.activation, 'sigmoid')
                    a = sigmoid(z);
                end
                
                if i < model.num_layers
                    a = [1; a];
                end
                
                outputs{i+1} = a;
            end
        end

        function model = backward(model, outputs, y)
            if strcmp(model.optimizer, 'sgd')
                delta = cell(1, model.num_layers);
                delta{model.num_layers} = sigmoid_derivative(outputs{end}) .* (outputs{end} - y);

                for i = model.num_layers-1 : -1 : 1
                    a = outputs{i+1}(2:end);
                    w = model.layers{i+1}.weights(:,2:end);
                    delta{i} = sigmoid_derivative(a) .* (w' * delta{i+1});
                end

                for i = 1 : model.num_layers
                    model.layers{i}.weights = model.layers{i}.weights - model.learning_rate * delta{i} * outputs{i}';
                end
            end
        end

        function [model, history] = train(model, X, Y, epochs)
            history = zeros(epochs, 1);
            [m, ~] = size(X);
            X = [ones(m, 1) X];
            for epoch = 1 : epochs
                loss = 0;
                for i = 1 : m
                    x = X(i, :)';
                    y = Y(i, :)';
                    outputs = forward(model, x);

                    if strcmp(model.loss_function, 'mse')
                        loss = loss + sum(0.5 * (outputs{end} - y).^2);
                    end

                    model = backward(model, outputs, y);
                end
                history(epoch) = loss / m;
            end
        end

        function y_pred = predict(model, X)
            [m, ~] = size(X);
            X = [ones(m, 1) X]';
            outputs = forward(model, X);
            y_pred = outputs{end};
        end
    end
end