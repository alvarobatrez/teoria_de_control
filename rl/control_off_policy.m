close all; clear, clc

R = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

start_position = [1 1];
[goal_row, goal_col] = find(R==0);

[m, n] = size(R);
num_actions = length(actions);

gamma = 0.99;
epsilon = 1;
decay = 0.99;
num_episodes = 1000;

Q = -100 * ones(m, n, num_actions);
Q(goal_row, goal_col, :) = 0;
Q(repmat(R == 1, 1, 1, size(Q, 3))) = 0;
C = zeros(m, n, num_actions);
[~, pi] = max(Q, [], 3);
pi(R==1) = 0;
pi(R==0) = 0;

u = ones(m, n, num_actions) / num_actions;

for episode = 1 : num_episodes
    [states, actions_taken, rewards] = generate_episode(R, u, start_position, [goal_row, goal_col], actions, num_actions, m, n);
    G = 0;
    W = 1;

    for t = length(states) : -1 : 1
        G = gamma * G + rewards(t);

        index = sub2ind([m, n, num_actions], states(t,1), states(t,2), actions_taken(t));

        C(index) = C(index) + W;
        Q(index) = Q(index) + W / C(index) * (G - Q(index));

        [~, A] = max(Q(states(t,1), states(t,2), :));
        pi(states(t,1), states(t,2)) = A;

        if actions_taken(t) ~= pi(states(t,1), states(t,2))
            break
        end

        W = W / u(states(t, 1), states(t, 2), actions_taken(t));

    end

    epsilon = max(0.1, decay*epsilon);
    u(states(t,1), states(t,2), :) = epsilon / num_actions;
    u(states(t,1), states(t,2), A) = 1 - epsilon + epsilon / num_actions;
end

policy = pi;

clc
disp('Acciones: 1=arriba, 2=derecha, 3=abajo, 4=izquierda')
disp('Politica Optima')
disp(policy)

plot_q_values(Q)

[row, col] = find(R == 0);
draw_maze(R, start_position, policy, [row col])