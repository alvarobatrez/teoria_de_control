function [states, actions_taken, rewards] = generate_episode(R, pi, start_position, goal_position, actions, num_actions, m, n)

state = start_position;
i = state(1);
j = state(2);
states = [];
actions_taken = [];
rewards = [];

while ~isequal(state, goal_position)

    states = [states; state];

    action_probabilities = squeeze(pi(i, j, :));
    action = randsample(1:num_actions, 1, true, action_probabilities);

    new_i = i + actions(action, 1);
    new_j = j + actions(action, 2);

    if new_i >= 1 && new_i <= m && new_j >=1 && new_j <= n && R(new_i, new_j) ~= 1
        i = new_i;
        j = new_j;
    end

    state = [i j];
    actions_taken = [actions_taken; action];
    rewards = [rewards; R(i, j)];
end