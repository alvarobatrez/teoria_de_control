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
N = 5;

Q = zeros(m, n, num_actions);

for episode = 1 : num_episodes
    epsilon = max(0.1, decay*epsilon);
    state = start_position;
    action = egreedy_action(epsilon, Q, state, num_actions);

    states = zeros(N, 2);
    actions_taken = zeros(N, 1);
    rewards = zeros(N, 1);

    states(1, :) = state;
    actions_taken(1) = action;

    T = inf;
    t = 1;

    while true
        if t < T
            [next_state, reward] = step(R, state, action, actions, m, n);
            states(t+1, :) = next_state;
            rewards(t+1) = reward;

            if isequal(next_state, [goal_row, goal_col])
                T = t + 1;
            else
                next_action = egreedy_action(epsilon, Q, next_state, num_actions);
                actions_taken(t+1) = next_action;
            end
            
            state = next_state;
            action = next_action;
        end

        tau = t - N + 1;

        if tau >= 1
            G = 0;
            for i = tau + 1 : min(tau + N, T)
                G = G + gamma^(i - tau - 1) * rewards(i);
            end

            if tau + N < T
                G = G + gamma^N * Q(states(tau+N, 1), states(tau+N, 2), actions_taken(tau+N, 1));
            end

            Q(states(tau, 1), states(tau, 2), actions_taken(tau, 1)) = ...
                Q(states(tau, 1), states(tau, 2), actions_taken(tau, 1)) + ...
                alpha * (G - Q(states(tau, 1), states(tau, 2), actions_taken(tau, 1)));
        end

        t = t + 1;

        if tau == T - 1
            break
        end
    end
    
    while ~isequal(state, [goal_row, goal_col])
        
        next_action = egreedy_action(epsilon, Q, next_state, num_actions);
        Q(state(1), state(2), action) = Q(state(1), state(2), action) + alpha * (reward + gamma * Q(next_state(1), next_state(2), next_action) - Q(state(1), state(2), action));
        
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