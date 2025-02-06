close all; clear, clc

M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

start_position = [1 2];
[goal_row, goal_col] = find(M==10);

[m, n] = size(M);
num_actions = length(actions);

gamma = 0.99;
epsilon = 1;
decay = 0.99;
num_episodes = 1000;

pi = ones(m, n, num_actions) / num_actions;
Q = zeros(m, n, num_actions);
N = zeros(m, n, num_actions);

for episode = 1 : num_episodes
    epsilon = max(0.1, decay*epsilon);
    [states, actions_taken, rewards] = generate_episode(M, pi, start_position, [goal_row, goal_col], actions, num_actions, m, n);
    G = 0;

    visited = false(m, n, num_actions);

    for t = length(states) : -1 : 1
        G = gamma * G + rewards(t);

        index = sub2ind([m, n, num_actions], states(t,1), states(t,2), actions_taken(t));

        if ~visited(index)
            visited(index) = true;

            N(index) = N(index) + 1;
            Q(index) = Q(index) + 1 / N(index) * (G - Q(index));

            [~, A] = max(Q(states(t, 1), states(t, 2), :));
            pi(states(t, 1), states(t, 2), :) = epsilon / num_actions;
            pi(states(t, 1), states(t, 2), A) = 1 - epsilon + epsilon / num_actions;
        end
    end
    fprintf('Episodio: %d\n', episode)
end

[~, policy] = max(Q, [], 3);
policy(M==-2) = 0;
policy(M==10) = 0;

plot_q_values(Q)

draw_maze(M, start_position, policy, [goal_col goal_row])