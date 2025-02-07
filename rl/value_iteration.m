close all; clear, clc

M = create_maze();
[goal_row, goal_col] = find(M==10);
actions = [-1 0; 0 1; 1 0; 0 -1];

[m, n] = size(M);
num_actions = length(actions);

theta = 1e-6;
gamma = 0.99;

V = zeros(m, n);
policy = zeros(m, n);

while true
    delta = 0;

    for i = 1 : m
        for j = 1 : n

            if M(i, j) == -2 || M(i, j) == 10
                continue
            end

            v = V(i, j);
            aux = zeros(1, num_actions);

            for action = 1 : num_actions

                new_i = i + actions(action, 1);
                new_j = j + actions(action, 2);

                if new_i < 1 || new_i > m || new_j < 1 || new_j > n || M(new_i, new_j) == -2
                    reward = -2;
                    new_i = i;
                    new_j = j;
                else
                    reward = M(new_i, new_j);
                end

                aux(action) = reward + gamma * V(new_i, new_j);
            end

            [V(i, j), policy(i, j)] = max(aux);
            delta = max(delta, abs(v - V(i, j)));
        end
    end

    if delta < theta
        break
    end
end

disp('Acciones: 1=arriba, 2=derecha, 3=abajo, 4=izquierda')
disp('Politica Optima')
disp(policy)

V(M==10) = 10;
draw_heatmap(V)

start_position = [1 2];
draw_maze(M, start_position, policy, [goal_row goal_col])