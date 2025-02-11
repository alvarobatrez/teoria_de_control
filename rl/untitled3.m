close all; clear, clc

J = 3.4e-5;
b = 2.2e-5;
K = 50e-3;
L = 7.7e-3;
R = 11.4;

alpha = 0.1;
gamma = 0.99;
epsilon = 1;
decay = 0.99;
num_episodes = 3000;

bins_w = 50;
bins_v = 50;
low = [0 0];
high = [20 1];
n = 4;

[tilings_w, tilings_v] = create_tilings([bins_w bins_v], low, high, n);

actions = [-0.01 0 0.01];
num_actions = length(actions);

Q = zeros(n, bins_w, bins_v, num_actions);

ref = 10;
Ts = 0.01;
t_steps = 100;

sum_returns = zeros(num_episodes, 1);

for episode = 1 : num_episodes
    w = 0;
    v = 0;
    i = 0;

    epsilon = max(0.001, decay*epsilon);
    state = discretize_state([w v], tilings_w, tilings_v);
    action = egreedy_action(epsilon, Q, state, num_actions, n);

    G = 0;

    for t = 1 : t_steps
        v = clamp_voltage(v + actions(action), [low(2) high(2)]);
        
        [next_w, next_i] = simulate_motor([w, i], Ts, J, b, K, L, R, v);
        
        reward = -(ref - next_w)^2;

        next_state = discretize_state([next_w v], tilings_w, tilings_v);
        next_action = egreedy_action(epsilon, Q, next_state, num_actions, n);

        for i = 1 : n
            Q(i, state(i,1), state(i,2), action) = Q(i, state(i,1), state(i,2), action) + alpha * (reward + gamma * Q(i, next_state(i,1), next_state(i,2), next_action) - Q(i, state(i,1), state(i,2), action));
        end
        
        w = next_w;
        i = next_i;
        state = next_state;
        action = next_action;

        G = G + reward;
    end

    sum_returns(episode) = G;

    fprintf('Episodio: %d\n', episode)
end

plot(1:num_episodes, sum_returns)
xlabel('Episodes'), ylabel('Returns'), title('Agregacion de Estados')

w = 0;
v = 0;
i = 0;
ref = 10;

draw_response(Q, actions, num_actions, n, [w v i], tilings_w, tilings_v, low, high, t_steps, Ts, J, b, K, L, R)

function [tilings_w, tilings_v] = create_tilings(bins, low, high, n)
    displacement_vector = 1 : 2 : 2*length(bins);
    tilings_w = zeros(n, bins(1));
    tilings_v = zeros(n, bins(2));
    for i = 1 : n
        low_i = low - rand * 0.2 * low;
        high_i = high + rand * 0.2 * high;
        segment_sizes = (high_i - low_i) ./ bins;
        displacements = mod(displacement_vector*i, n);
        displacements = displacements .* (segment_sizes / n);
        low_i = low_i + displacements;
        high_i = high_i + displacements;
        w_range = linspace(low_i(1), high_i(1), bins(1));
        v_range = linspace(low_i(2), high_i(2), bins(2));
        tilings_w(i, :) = w_range;
        tilings_v(i, :) = v_range;
    end
end

function states = discretize_state(observations, tilings_w, tilings_v)
    w = observations(1);
    v = observations(2);
    [~, w_index] = min(abs(tilings_w - w), [], 2);
    [~, v_index] = min(abs(tilings_v - v), [], 2);
    states = [w_index v_index];
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

function v = clamp_voltage(v, v_range)
    v = max(v_range(1), min(v, v_range(end)));
end

function [next_w, next_i] = simulate_motor(state, Ts, J, b, K, L, R, v)
    fun = @(t,x) motor(x, J, b, K, L, R, v);
    [~, x] = ode45(fun, [0 Ts], state);
    next_w = x(end, 1);
    next_i = x(end, 2);
end

function dx = motor(x, J, b, K, L, R, v)
    dx1 = -b/J*x(1) + K/J*x(2);
    dx2 = -K/L*x(1) - R/L*x(2) + v/L;
    dx = [dx1; dx2];
end

function draw_response(Q, actions, num_actions, n, initial_conditions, w_range, v_range, low, high, t_steps, Ts, J, b, K, L, R)
    w = initial_conditions(1);
    v = initial_conditions(2);
    i = initial_conditions(3);
    state = discretize_state([w v], w_range, v_range);

    omega = zeros(t_steps, 1);

    for t = 1 : t_steps
        
        q = zeros(num_actions, 1);
        for a = 1 : num_actions
            for j = 1 : n
                q(a) = q(a) + Q(j, state(j,1), state(j,2), a);
            end
        end

        [~, action] = max(q);

        v = clamp_voltage(v + actions(action), [low(2) high(2)]);
        [w, i] = simulate_motor([w, i], Ts, J, b, K, L, R, v);
        state = discretize_state([w v], w_range, v_range);

        omega(t) = w;
    end

    t = (0:t_steps-1)*Ts;
    figure, plot(t, omega)
    xlabel('Tiempo'), ylabel('Velocidad angular'), title('Motor DC')
end