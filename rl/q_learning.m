close all; clear, clc

R = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

start_position = [1 1];
[goal_row, goal_col] = find(R==0);

[m, n] = size(R);
num_actions = length(actions);

alpha = 0.1;
gamma = 0.99;
epsilon = 1;
decay = 0.99;
num_episodes = 1000;

Q = zeros(m, n, num_actions);

for episode = 1 : num_episodes
    epsilon = max(0.1, decay*epsilon);
    state = start_position;
    
    while ~isequal(state, [goal_row, goal_col])
        action = egreedy_action(epsilon, Q, state, num_actions);
        [next_state, reward] = step(R, state, action, actions, m, n);
        Q(state(1), state(2), action) = Q(state(1), state(2), action) + alpha * (reward + gamma * max(Q(next_state(1), next_state(2), :)) - Q(state(1), state(2), action));
        state = next_state;
    end
end

[~, policy] = max(Q, [], 3);
policy(R==1) = 0;
policy(R==0) = 0;

clc
disp('Acciones: 1=arriba, 2=derecha, 3=abajo, 4=izquierda')
disp('Politica Optima')
disp(policy)

plot_q_values(Q)

[row, col] = find(R == 0);
draw_maze(R, start_position, policy, [row col])

function action = egreedy_action(epsilon, Q, state, num_actions)
    if rand > epsilon
        [~, action] = max(Q(state(1), state(2), :));
    else
        action = randi(num_actions);
    end
end

function [next_state, reward] = step(R, state, action, actions, m, n)
    i = state(1);
    j = state(2);

    new_i = i + actions(action, 1);
    new_j = j + actions(action, 2);

    if new_i >= 1 && new_i <= m && new_j >=1 && new_j <= n && R(new_i, new_j) ~= 1
        i = new_i;
        j = new_j;
    end

    next_state = [i j];
    reward = R(i, j);
end