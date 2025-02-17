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
num_episodes = 1000;
max_steps = 1e5;

buffer_capacity = 1e5;
batch_size = 64;
buffer = ExperienceReplay(buffer_capacity);

num_inputs = 2;
learning_rate = 0.001;
optimizer = 'adamW';
loss_function = 'mse';

layers = {{128, 'relu'} {64, 'relu'} {4, 'linear'}};

model = NeuralNetwork(num_inputs, layers);
model = model.compile(learning_rate, optimizer, loss_function);

model_target = NeuralNetwork(num_inputs, layers);
model_target = model_target.compile(learning_rate, optimizer, loss_function);
model_target = copy_weights(model, model_target);

for episode = 1 : num_episodes
    epsilon = max(0.1, decay*epsilon);
    state = start_position;
    steps = 0;
    
    while ~isequal(state, [goal_row goal_col]) && steps < max_steps
        steps = steps + 1;
        action = egreedy_action(epsilon, model, state, num_actions);
        [next_state, reward] = step(M, state, action, actions, m, n);
        buffer = buffer.insert([state action reward next_state]);
        
        if buffer.can_sample(batch_size) == true
            sample = buffer.sample(batch_size);
            [state_b, action_b, reward_b, next_state_b] = split_sample(sample, batch_size, num_inputs);
            q_b = gather_q(model, state_b, action_b, batch_size);
            next_action_b = egreedy_action(epsilon, model, next_state_b, num_actions);
            next_q_b = gather_q(model_target, next_state_b, next_action_b, batch_size);
            done = zeros(batch_size, 1);
            ind = find(all(next_state_b == [goal_row goal_col], 2));
            done(ind) = 1;
            target_b = reward_b + ~done .* gamma .* next_q_b;

            predictions = model.predict(state_b);
            target_q = predictions;

            for i = 1 : batch_size
                target_q(i, action_b(i)) = target_b(i);
            end

            model = backpropagation(model, state_b, target_q, batch_size);
        end

        state = next_state;
    end

    for i = 1 : model.num_layers
        model_target.layers{i}.weights = tau * model.layers{i}.weights + (1 - tau) * model_target.layers{i}.weights;
    end
    
    fprintf('Episodio: %d, Pasos: %d\n', episode, steps)
end

policy = create_policy(model, M);

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

function [state_b, action_b, reward_b, next_state_b] = split_sample(sample, batch_size, num_inputs)
    state_b = zeros(batch_size, num_inputs);
    action_b = zeros(batch_size, 1);
    reward_b = zeros(batch_size, 1);
    next_state_b = zeros(batch_size, num_inputs);

    for i = 1 : batch_size
        state_b(i, :) = sample{i}(1 : num_inputs);
        action_b(i, :) = sample{i}(num_inputs + 1);
        reward_b(i, :) = sample{i}(num_inputs + 2);
        next_state_b(i, :) = sample{i}(num_inputs + 3 : end);
    end
end

function q = gather_q(model, state, action, batch_size)
    predictions = model.predict(state);
    indices = sub2ind(size(predictions), (1 : batch_size)', action);
    q = predictions(indices);
end

function model = backpropagation(model, X, Y, batch_size)
    outputs = forward(model, X);
    grad = model.compute_gradients(batch_size, outputs, Y);
    model = model.update_weights(grad);
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