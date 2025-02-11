close all; clear, clc

J = 3.4e-5;
b = 2.2e-5;
K = 50e-3;
L = 7.7e-3;
R = 11.4;

v_min = 0; v_max = 1;
w_min = 0; w_max = 20;
bins_v = 50;
bins_w = 50;
low = [v_min w_min];
high = [v_max w_max];
n_tilings = 8;

[tilings_v, tilings_w] = create_tilings([bins_v bins_w], low, high, n_tilings);

actions = [-0.01 0 0.01];
num_actions = length(actions);

ref = 10;
Ts = 0.01;
T = 1;
t_steps = T / Ts;

alpha = 0.1 / n_tilings;
gamma = 0.99;
epsilon = 1;
decay = 0.995;
num_episodes = 2000;

Q = zeros(n_tilings, bins_v, bins_w, num_actions);

sum_returns = zeros(num_episodes, 1);

for episode = 1 : num_episodes
    epsilon = max(0.1, decay*epsilon);

    v = 0;
    w = 0;
    ic = 0;
    state = discretize_state([v w], tilings_v, tilings_w);

    G = 0;
    
    for t = 1 : t_steps
        action = egreedy_action(epsilon, Q, state, num_actions, n_tilings);
        v = clamp_voltage(v + actions(action), v_min, v_max);

        [w, ic] = simulate_motor([w ic], Ts, J, b, K, L, R, v);

        next_state = discretize_state([v w], tilings_v, tilings_w);
        next_action = egreedy_action(epsilon, Q, next_state, num_actions, n_tilings);

        reward = -(ref - w)^2;

        for i = 1 : n_tilings
            Q(i, state(i, 1), state(i, 2), action) = Q(i, state(i, 1), state(i, 2), action) + ...
            alpha * (reward + gamma * Q(i, next_state(i, 1), next_state(i, 2), next_action) - Q(i, state(i, 1), state(i, 2), action));
        end

        state = next_state;
        action = next_action;

        G = G + reward;
    end

    sum_returns(episode) = G;
    
    fprintf('Episodio: %d\n', episode)
end

q = squeeze(sum(Q, 1));
[~, policy] = max(q, [], 3);

initial_conditions = [0 0 0]; % v w i
draw_response(policy, actions, initial_conditions, tilings_v, tilings_w, v_min, v_max, t_steps, Ts, J, b, K, L, R)

figure, plot(1:num_episodes, sum_returns)
xlabel('Episodio'), ylabel('Retornos'), title('SARSA con Agregación de Estados')

t = (0 : t_steps-1) * Ts;

function [tilings_v, tilings_w] = create_tilings(bins, low, high, n)
    displacement_vector = 1 : 2 : 2*length(bins);
    tilings_v = zeros(n, bins(1));
    tilings_w = zeros(n, bins(2));
    for i = 1 : n
        low_i = low - rand * 0.2 * low;
        high_i = high + rand * 0.2 * high;
        segment_sizes = (high_i - low_i) ./ bins;
        displacements = mod(displacement_vector*i, n);
        displacements = displacements .* (segment_sizes / n);
        low_i = low_i + displacements;
        high_i = high_i + displacements;
        v_range = linspace(low_i(1), high_i(1), bins(1));
        w_range = linspace(low_i(2), high_i(2), bins(2));
        tilings_v(i, :) = v_range;
        tilings_w(i, :) = w_range;
    end
end

function state = discretize_state(observations, tilings_v, tilings_w)
    v = observations(1);
    w = observations(2);
    [~, v_index] = min(abs(tilings_v - v), [], 2);
    [~, w_index] = min(abs(tilings_w - w), [], 2);
    state = [v_index w_index];
end

function action = egreedy_action(epsilon, Q, state, num_actions, n)
    if rand > epsilon
        action_values = zeros(n, num_actions);
        for i = 1 : n
            av = Q(i, state(i,1), state(i, 2), :);
            av = reshape(av, [1, num_actions]);
            action_values(i,:) = av;
        end
        [~, action] = max(mean(action_values));
    else
        action = randi(num_actions);
    end
end

function v = clamp_voltage(v, v_min, v_max)
    v = max(v_min, min(v, v_max));
end

function dx = motor(x, J, b, K, L, R, v)
    dx1 = -b/J*x(1) + K/J*x(2);
    dx2 = -K/L*x(1) - R/L*x(2) + v/L;
    dx = [dx1; dx2];
end

function [w, i] = simulate_motor(state, Ts, J, b, K, L, R, v)
    fun = @(t,x) motor(x, J, b, K, L, R, v);
    [~, x] = ode23(fun, [0 Ts], state);
    w = x(end, 1);
    i = x(end, 2);
end

function draw_response(policy, actions, initial_conditions, tilings_v, tilings_w, v_min, v_max, t_steps, Ts, J, b, K, L, R)
    v = initial_conditions(1);
    w = initial_conditions(2);
    i = initial_conditions(3);

    state = discretize_state([v w], tilings_v, tilings_w);

    vel_motor = zeros(t_steps, 1);
    
    for t = 1 : t_steps
        action = policy(state(1), state(2));
        v = clamp_voltage(v + actions(action), v_min, v_max);
        [w, i] = simulate_motor([w i], Ts, J, b, K, L, R, v);
        state = discretize_state([v w], tilings_v, tilings_w);

        vel_motor(t) = w;
    end

    t = (0 : t_steps-1) * Ts;
    figure, plot(t, vel_motor)
    xlabel('Tiempo'), ylabel('Velocidad angular'), title('Motor DC')
end