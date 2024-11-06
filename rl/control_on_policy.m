close all; clear, clc

R = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

[m, n] = size(R);
num_actions = length(actions);

gamma = 0.99;
epsilon = 1;
decay = 0.99;
num_episodes = 1000;

pi = ones(m, n, num_actions) / num_actions;
Q = zeros(m, n, num_actions);
N = zeros(m, n, num_actions);

start_position = [1 1];
[goal_row, goal_col] = find(R==0);

for episode = 1 : num_episodes
    epsilon = max(0.1, decay*epsilon);
    [states, actions_taken, rewards] = generate_episode(R, pi, start_position, [goal_row, goal_col], actions, num_actions, m, n);
    sa = [states actions_taken];
    G = 0;
    
    for t = length(states) : -1 : 1
        G = gamma * G + rewards(t);

        if ~ismember(sa(t,:), sa(1:t-1,:), 'rows')
            index = sub2ind([m, n, num_actions], states(t,1), states(t,2), actions_taken(t));
            N(index) = N(index) + 1;
            Q(index) = Q(index) + 1 / N(index) * (G - Q(index));
            [~, A] = max(Q(states(t,1), states(t,2), :));

            pi(states(t,1), states(t,2), :) = epsilon / num_actions;
            pi(states(t,1), states(t,2), A) = 1 - epsilon + epsilon / num_actions;
        end
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