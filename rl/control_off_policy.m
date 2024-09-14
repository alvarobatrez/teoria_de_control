close all; clear, clc

R = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

[m, n] = size(R);
num_actions = length(actions);

start_position = [1 1];
[goal_row, goal_col] = find(R==0);

gamma = 0.99;
epsilon = 0.1;
num_episodes = 1000;

Q = -100 * ones(m, n, num_actions);
Q(goal_row, goal_col,:) = 0;
Q(repmat(R == 1, 1, 1, size(Q, 3))) = 0;
C = zeros(m, n, num_actions);
pi = ones(m, n, num_actions) / num_actions;

for episode = 1 : num_episodes
    b = pi;
    [states, actions_taken, rewards] = generate_episode(R, b, start_position, [goal_row, goal_col], actions, num_actions, episode, m, n);
    G = 0;
    W = 1;
    
    for t = length(states) : -1 : 1
        index = sub2ind([m, n, num_actions], states(t,1), states(t,2), actions_taken(t));

        G = gamma * G + rewards(t);
        C(index) = C(index) + W;
        Q(index) = Q(index) + W / C(index) * (G - Q(index));

        [~, A] = max(Q(states(t,1), states(t,2), :));
        for a = 1 : num_actions
            if a == A
                pi(states(t,1), states(t,2), a) = 1 - epsilon + epsilon / num_actions;
            else
                pi(states(t,1), states(t,2), a) = epsilon / num_actions;
            end
        end

        if actions_taken(t) ~= A
            break
        end

        W = W / b(states(t, 1), states(t, 2), actions_taken(t));
    end
end

[~, policy] = max(pi, [], 3);
policy(R==1) = 0;
policy(R==0) = 0;

clc
disp('Acciones: 1=arriba, 2=derecha, 3=abajo, 4=izquierda')
disp('Politica Optima')
disp(policy)

plot_q_values(Q)

start_position = [1 1];
[row, col] = find(R == 0);
draw_maze(R, start_position, policy, [row col])