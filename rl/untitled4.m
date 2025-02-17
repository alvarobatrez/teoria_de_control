close all; clear, clc

M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

start_position = [1 2];
[goal_row, goal_col] = find(M==10);

[m, n] = size(M);
num_actions = length(actions);

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

layers = {{64, 'relu'} {4, 'linear'}};

model = NeuralNetwork(num_inputs, layers);
model = model.compile(learning_rate, optimizer, loss_function);

model_target = NeuralNetwork(num_inputs, layers);
model_target = model_target.compile(learning_rate, optimizer, loss_function);
model_target = copy_weights(model, model_target);

tau = 0.01;  % Para soft-update (opcional, mejora estabilidad)

for episode = 1 : num_episodes
    epsilon = max(0.1, decay*epsilon);
    state = start_position;
    steps = 0;
    
    while ~isequal(state, [goal_row goal_col]) && steps < max_steps
        steps = steps + 1;
        % 1. Selección de acción on-policy
        action = egreedy_action(epsilon, model, state, num_actions);
        
        % 2. Interacción con el entorno
        [next_state, reward] = step(M, state, action, actions, m, n);
        
        % 3. Selección de next_action usando política actual (SARSA)
        next_action = egreedy_action(epsilon, model, next_state, num_actions);
        
        % 4. Cálculo de Q-values (sin buffer)
        q_current = gather_q(model, state, action, 1);  % batch_size=1
        q_next = gather_q(model_target, next_state, next_action, 1);
        
        % 5. Detección de estado terminal
        done = isequal(next_state, [goal_row goal_col]);
        target = reward + ~done * gamma * q_next;
        
        % 6. Actualización online de la red
        predictions = model.predict(state);
        predictions(action) = target;
        model = backpropagation(model, state, predictions, 1);  % batch_size=1
        
        % 7. Soft-update del target network (opcional pero recomendado)
        %model_target = update_target(model, model_target, tau);
        
        state = next_state;
    end

    if mod(episode, 10) == 0
        model_target = copy_weights(model, model_target);
    end
    
    fprintf('Episodio: %d, Pasos: %d\n', episode, steps)
end

policy = create_policy(model, M);
draw_maze(M, start_position, policy, [goal_row goal_col])

% --------------------- Funciones auxiliares ---------------------
function model_target = update_target(model, model_target, tau)
    for i = 1 : model.num_layers
        model_target.layers{i}.weights = tau * model.layers{i}.weights + ...
                                      (1 - tau) * model_target.layers{i}.weights;
    end
end

function action = egreedy_action(epsilon, model, state, num_actions)
    if rand > epsilon
        [~, action] = max(model.predict(state));
    else
        action = randi(num_actions);
    end
end

function q = gather_q(model, state, action, ~)
    predictions = model.predict(state);
    q = predictions(action);
end

function model = backpropagation(model, X, Y, ~)
    outputs = forward(model, X);
    grad = model.compute_gradients(1, outputs, Y);  % batch_size=1
    model = model.update_weights(grad);
end

function model_copy = copy_weights(model_original, model_copy)
    for i = 1 : model_original.num_layers
        model_copy.layers{i}.weights = model_original.layers{i}.weights;
    end
end