classdef NeuralNetwork

    properties
        layers
        learning_rate
        optimizer
        loss_function
        num_layers
        beta1
        beta2
        epsilon
        m
        v
    end

    methods
        function model = NeuralNetwork(input_size, layers)
            model.num_layers = length(layers);
            model.beta1 = 0.9;
            model.beta2 = 0.999;
            model.epsilon = 1e-8;
            model.m = cell(1, model.num_layers);
            model.v = cell(1, model.num_layers);

            for i = 1 : model.num_layers
                if i == 1
                    weights = randn(layers{i}{1}, input_size + 1) * sqrt(2 / input_size);
                else
                    weights = randn(layers{i}{1}, layers{i-1}{1} + 1) * sqrt(2 / layers{i-1}{1});
                end

                model.layers{i} = struct('weights', weights, 'activation', layers{i}{2});
                model.m{i} = zeros(size(weights));
                model.v{i} = zeros(size(weights));
            end
        end

        function model = compile(model, learning_rate, optimizer, loss_function)
            model.learning_rate = learning_rate;
            model.optimizer = optimizer;
            model.loss_function = loss_function;
        end

        function outputs = forward(model, x)
            outputs = cell(1, model.num_layers + 1);
            outputs{1} = x;

            for i = 1 : model.num_layers
                z = model.layers{i}.weights * outputs{i}';
                activation_function = model.layers{i}.activation;

                if strcmp(activation_function, 'sigmoid')
                    a = sigmoid(z);
                elseif strcmp(activation_function, 'tanh')
                    a = tanh(z);
                elseif strcmp(activation_function, 'relu')
                    a = relu(z);
                elseif strcmp(activation_function, 'linear')
                    a = z;
                end

                if i < model.num_layers
                    outputs{i+1} = [ones(1, size(a, 2)); a]';
                else
                    outputs{i+1} = a';
                end
            end
        end

        function derivative = activation_derivative(~, layer_activation, x)
            switch layer_activation
                case 'sigmoid'
                    derivative = sigmoid_derivative(x);
                case 'tanh'
                    derivative = tanh_derivative(x);
                case 'relu'
                    derivative = relu_derivative(x);
                case 'linear'
                    derivative = 1;
            end
        end

        function grad = compute_gradients(model, batch_size, outputs, y)
            delta = cell(1, model.num_layers);
            layer_activation = model.layers{model.num_layers}.activation;
            grad = cell(1, model.num_layers);

            derivative = model.activation_derivative(layer_activation, outputs{end});

            delta{model.num_layers} = derivative .* (outputs{end} - y);

            for i = model.num_layers - 1 : -1 : 1
                layer_activation = model.layers{i}.activation;
                a = outputs{i+1}(:, 2:end);

                derivative = model.activation_derivative(layer_activation, a);
                
                w = model.layers{i+1}.weights(:, 2:end);
                delta{i} = derivative .* (delta{i + 1} * w);
            end

            for i = 1 : model.num_layers
                grad{i} = (1 / batch_size) * delta{i}' * outputs{i};
            end
        end

        function model = update_weights(model, grad, epoch)
            for i = 1 : model.num_layers
                if strcmp(model.optimizer, 'sgd')
                    model.layers{i}.weights = model.layers{i}.weights - model.learning_rate * grad{i};
                elseif strcmp(model.optimizer, 'adam')
                    model.m{i} = model.beta1 * model.m{i} + (1 - model.beta1) * grad{i};
                    model.v{i} = model.beta2 * model.v{i} + (1 - model.beta2) * (grad{i}.^2);
                    m_hat = model.m{i} / (1 - model.beta1^epoch);
                    v_hat = model.v{i} / (1 - model.beta2^epoch);
                    model.layers{i}.weights = model.layers{i}.weights - model.learning_rate * m_hat ./ (sqrt(v_hat) + model.epsilon);
                end
            end
        end

        function [model, history] = train(model, X, Y, epochs)
            history = zeros(epochs, 1);
            [batch_size, ~] = size(X);
            [~, num_outputs] = size(Y);
            X = [ones(batch_size, 1) X];

            for epoch = 1 : epochs
                loss = 0;
                outputs = forward(model, X);

                if strcmp(model.loss_function, 'mse')
                    loss = loss + sum((outputs{end} - Y).^2, 'all') / num_outputs;
                end

                grad = compute_gradients(model, batch_size, outputs, Y);
                model = update_weights(model, grad, epoch); 
                
                history(epoch) = loss;
            end
        end

        function y_pred = predict(model, X)
            x = [ones(size(X, 1), 1) X];
            outputs = forward(model, x);
            y_pred = outputs{end};
        end
    end
end