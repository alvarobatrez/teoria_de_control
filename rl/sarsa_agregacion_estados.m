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
w_range = linspace(0, 20, bins_w);
v_range = linspace(0, 1, bins_v);

actions = [-0.01 0 0.01];
num_actions = length(actions);

Q = zeros(bins_w, bins_v, num_actions);

ref = 10;
Ts = 0.01;
t_steps = 100;

sum_returns = zeros(num_episodes, 1);

for episode = 1 : num_episodes
    w = 0;
    v = 0;
    i = 0;

    epsilon = max(0.001, decay*epsilon);
    state = discretize_state([w v], w_range, v_range);
    action = egreedy_action(epsilon, Q, state, num_actions);

    G = 0;

    for t = 1 : t_steps
        v = clamp_voltage(v + actions(action), v_range);
        
        [next_w, next_i] = simulate_motor([w, i], Ts, J, b, K, L, R, v);
        
        reward = -(ref - next_w)^2;

        next_state = discretize_state([next_w v], w_range, v_range);
        next_action = egreedy_action(epsilon, Q, next_state, num_actions);

        Q(state(1), state(2), action) = Q(state(1), state(2), action) + alpha * (reward + gamma * Q(next_state(1), next_state(2), next_action) - Q(state(1), state(2), action));
        
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

[~, policy] = max(Q, [], 3);
draw_response(policy, actions, [w v i], w_range, v_range, t_steps, Ts, J, b, K, L, R)

function state = discretize_state(observations, w_range, v_range)
    w = observations(1);
    v = observations(2);
    [~, w_index] = min(abs(w_range - w));
    [~, v_index] = min(abs(v_range - v));
    state = [w_index v_index];
end

function action = egreedy_action(epsilon, Q, state, num_actions)
    if rand > epsilon
        [~, action] = max(Q(state(1), state(2), :));
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

function draw_response(policy, actions, initial_conditions, w_range, v_range, t_steps, Ts, J, b, K, L, R)
    w = initial_conditions(1);
    v = initial_conditions(2);
    i = initial_conditions(3);
    state = discretize_state([w v], w_range, v_range);

    omega = zeros(t_steps, 1);

    for t = 1 : t_steps
        action = policy(state(1), state(2));
        v = clamp_voltage(v + actions(action), v_range);
        [w, i] = simulate_motor([w, i], Ts, J, b, K, L, R, v);
        state = discretize_state([w v], w_range, v_range);

        omega(t) = w;
    end

    t = (0:t_steps-1)*Ts;
    figure, plot(t, omega)
    xlabel('Tiempo'), ylabel('Velocidad angular'), title('Motor DC')
end