close all; clear, clc

M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

start_position = [1 2];
[goal_row, goal_col] = find(M==10);

[m, n] = size(M);
num_actions = length(actions);

tau = 0.005;
gamma = 0.99;
epsilon = 1;
decay = 0.995;
num_episodes = 2500;
max_steps = 1e5;

buffer_capacity = 1e6;
batch_size = 32;
buffer = ExperienceReplay(buffer_capacity);

num_inputs = 2;
layers = {{128, 'relu'} {32, 'relu'} {8, 'relu'} {4, 'linear'}};

learning_rate = 0.001;
optimizer = 'adam';
loss_function = 'mse';

q_network = NeuralNetwork(num_inputs, layers);
q_network = q_network.compile(learning_rate, optimizer, loss_function);

target_network = NeuralNetwork(num_inputs, layers);
target_network = target_network.compile(learning_rate, optimizer, loss_function);
target_network = copy_weights(q_network, target_network);

total_loss = zeros(num_episodes, 1);

for episode = 1 : num_episodes
    epsilon = max(0.1, decay*epsilon);
    state = start_position;
    steps = 0;
    loss = 0;
    n = 0;
    
    while ~isequal(state, [goal_row goal_col]) && steps < max_steps
        steps = steps + 1;
        action = egreedy_action(epsilon, q_network, state, num_actions);
        [next_state, reward, done] = step(M, state, action, actions, m, n);

        buffer = buffer.insert([state action reward done next_state]);

        if buffer.can_sample(batch_size) == true
            sample = buffer.sample(batch_size);
            [state_b, action_b, reward_b, done_b, next_state_b] = split_sample(sample);
            
            next_action_b = egreedy_action(epsilon, q_network, next_state_b, num_actions);
            next_q_b = gather_q(target_network, next_state_b, next_action_b, batch_size);
            target_b = reward_b + (1 - done_b) * gamma .* next_q_b;

            q_b = gather_q(q_network, state_b, action_b, batch_size);
            q_network = backpropagation(q_network, batch_size, state_b, target_b, action_b);

            target_network = update_target_network(q_network, target_network, tau);
            
            n = n + 1;
            loss = loss + 1 / n * (mean((target_b - q_b).^2) - loss);
        end        
        
        state = next_state;
    end

    % if mod(episode, 10) == 0
    %     target_network = copy_weights(q_network, target_network);
    % end

    total_loss(episode) = loss;
    
    fprintf('Episodio: %d, Pasos: %d, Pérdida: %.4f\n', episode, steps, loss)
end

policy = create_policy(q_network, M);

plot(1:num_episodes, total_loss), grid on
title('Deep Sarsa'), xlabel('Épocas'), ylabel('Error (MSE)')

draw_maze(M, start_position, policy, [goal_row goal_col])

function model_copy = copy_weights(model_original, model_copy)
    for i = 1 : model_original.num_layers
        model_copy.layers{i}.weights = model_original.layers{i}.weights;
    end
end

function action = egreedy_action(epsilon, model, state, num_actions)
    if rand > epsilon
        [~, action] = max(model.predict(state), [], 2);
    else
        [m, ~] = size(state);
        action = randi(num_actions, [m 1]);
    end
end

function [state_b, action_b, reward_b, done_b, next_state_b] = split_sample(sample)
    sample_reshaped = reshape(cell2mat(sample), 7, [])';
    state_b = sample_reshaped(:, 1:2);
    action_b = sample_reshaped(:, 3);
    reward_b = sample_reshaped(:, 4);
    done_b = sample_reshaped(:, 5);
    next_state_b = sample_reshaped(:, 6:7);
end

function q = gather_q(model, state, action, batch_size)
    q_values = model.predict(state);
    indices = sub2ind(size(q_values), (1 : batch_size)', action);
    q = q_values(indices);
end

function grad = compute_loss_gradients(model, batch_size, state, target, action)
    delta = cell(1, model.num_layers);
    outputs = model.forward(state);

    q_values = outputs{end};
    indices = sub2ind(size(q_values), (1 : batch_size)', action);
    q_values(indices) = target;

    delta{end} = 2 * (outputs{end} - q_values) / batch_size;

    for i = model.num_layers - 1 : -1 : 1
        layer_activation = model.layers{i}.activation;
        a = outputs{i+1};
        derivative = model.activation_derivative(layer_activation, a);
        
        w = model.layers{i + 1}.weights(:, 2:end);
        delta{i} = derivative .* (delta{i + 1} * w);
    end

    grad = cell(1, model.num_layers);

    for i = 1 : model.num_layers
        grad{i} = (1 / batch_size) * delta{i}' * [ones(batch_size, 1), outputs{i}];
    end
end

function model = backpropagation(model, batch_size, state, target, action)
    grad = compute_loss_gradients(model, batch_size, state, target, action);
    model = model.update_weights(grad);
end

function model_target = update_target_network(model, model_target, tau)
    for i = 1 : model.num_layers
        model_target.layers{i}.weights = tau * model.layers{i}.weights + (1 - tau) * model_target.layers{i}.weights;
    end
end

function policy = create_policy(model, M)
    [m, n] = size(M);
    policy = zeros(m, n);

    for i = 1 : m
        for j = 1 : n
            if M(i, j) == -1
                [~, action] = max(model.predict([i j]));
                policy(i, j) = action;
            end
        end
    end
end