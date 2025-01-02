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