close all; clear, clc

R = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

start_position = [1 1];
[goal_row, goal_col] = find(R==0);
R(goal_row, goal_col) = 10;

[m, n] = size(R);
num_actions = length(actions);

gamma = 0.9;
epsilon = 1;
decay = 0.999;
num_episodes = 2000;
max_steps = 10000;

buffer_capacity = 1e4;
batch_size = 64;
buffer = ExperienceReplay(buffer_capacity);

num_inputs = 2;
learning_rate = 0.001;
optimizer = 'adam';
loss_function = 'mse';

layers = {{32, 'relu'} {4, 'linear'}};

model = NeuralNetwork(num_inputs, layers);
model = model.compile(learning_rate, optimizer, loss_function);

model_target = NeuralNetwork(num_inputs, layers);
model_target = model_target.compile(learning_rate, optimizer, loss_function);

for i = 1 : model.num_layers
    model_target.layers{i}.weights = model.layers{i}.weights;
end

counter = 1;
returns = zeros(num_episodes, 1);

for episode = 1 : num_episodes
    steps = 0;
    loss = 0;
    episode_returns = 0;
    epsilon = max(0.1, decay*epsilon);
    state = start_position;
    
    while ~isequal(state, [goal_row, goal_col]) && steps < max_steps
        steps = steps + 1;
        action = egreedy_action(epsilon, model, state, num_actions);
        [next_state, reward] = step(R, state, action, actions, m, n);
        buffer = buffer.insert([state action reward next_state]);

        if buffer.can_sample(batch_size) == true
            sample = buffer.sample(batch_size);
            [state_b, action_b, reward_b, next_state_b] = split_sample(sample, batch_size, num_inputs);
            q_b = gather_q(model, state_b, action_b, batch_size);
            next_action_b = egreedy_action(epsilon, model, next_state_b, num_actions);
            next_q_b = gather_q(model_target, next_state_b, next_action_b, batch_size);
            done = zeros(batch_size, 1);
            ind = find(all(state_b == [goal_row, goal_col], 2));
            done(ind) = 1;
            target_b = reward_b + ~done .* gamma .* next_q_b;
            
            loss = loss + sum((q_b - target_b).^2) / batch_size;
            
            predictions = model.predict(state_b);
            target_q = predictions;
            
            for i = 1 : batch_size
                target_q(i, action_b(i)) = target_b(i);
            end

            model = backpropagation(model, state_b, target_q, batch_size, counter);
            counter = counter + 1;
        end
        
        state = next_state;
        episode_returns = episode_returns + reward;
    end

    if mod(episode, 5) == 0
        for i = 1 : model.num_layers
            model_target.layers{i}.weights = model.layers{i}.weights;
        end
    end

    returns(episode) = episode_returns;
    fprintf('Episode: %d\t Loss: %.3f\t Return: %d\n', episode, loss, episode_returns)
end

policy = create_policy(model, R);

clc
disp('Acciones: 1=arriba, 2=derecha, 3=abajo, 4=izquierda')
disp('Politica Optima')
disp(policy)

plot(1:num_episodes, returns), title('Deep Sarsa')
xlabel('Episodio'), ylabel('Retornos')
R(goal_row, goal_col) = 0;
draw_maze(R, start_position, policy, [goal_row goal_col])

function action = egreedy_action(epsilon, model, state, num_actions)
    [m, ~] = size(state);
    
    if rand > epsilon
        [~, action] = max(model.predict(state), [], 2);
    else
        action = randi(num_actions, [m, 1]);
    end
end

function [state_b, action_b, reward_b, next_state_b] = split_sample(sample, batch_size, num_inputs)
    state_b = zeros(batch_size, num_inputs);
    action_b = zeros(batch_size, 1);
    reward_b = zeros(batch_size, 1);
    next_state_b = zeros(batch_size, num_inputs);

    for i = 1 : batch_size
        state_b(i,:) = sample{i}(1:num_inputs);
        action_b(i,:) = sample{i}(num_inputs+1);
        reward_b(i,:) = sample{i}(num_inputs+2);
        next_state_b(i,:) = sample{i}(num_inputs+3:end);
    end
end

function model = backpropagation(model, X, Y, batch_size, counter)
    X = [ones(batch_size, 1) X];
    outputs = forward(model, X);
    grad = compute_gradients(model, batch_size, outputs, Y);
    model = update_weights(model, grad, counter);
end

function q = gather_q(model, state, action, batch_size)
    predictions = model.predict(state);
    indices = sub2ind(size(predictions), (1:batch_size)', action);
    q = predictions(indices);
end

function policy = create_policy(model, R)
    [m, n] = size(R);
    policy = zeros(m, n);
    
    for i = 1 : m
        for j = 1 : n
            if R(i,j) == -1
                [~, action] = max(model.predict([i j]));
                policy(i,j) = action;
            end
        end
    end
end