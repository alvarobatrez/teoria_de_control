function [next_state, reward] = step(R, state, action, actions, m, n)
i = state(1);
j = state(2);

new_i = i + actions(action, 1);
new_j = j + actions(action, 2);

if new_i >= 1 && new_i <= m && new_j >=1 && new_j <= n && R(new_i, new_j) ~= 1
    i = new_i;
    j = new_j;
    next_state = [i j];
    reward = R(i, j);
else
    next_state = [i j];
    reward = -2;
end

