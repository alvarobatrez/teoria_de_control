close all; clear, clc

R = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

[m, n] = size(R);
num_actions = length(actions);

theta = 1e-6;
gamma = 0.9;

V = zeros(m, n);
pi = zeros(m, n);

while true
    delta = 0;
    
    for i = 1 : m
        for j = 1 : n

            if R(i, j) == 0 || R(i, j) == 1
                continue
            end

            v = V(i, j);
            aux = zeros(1, num_actions);

            for action = 1 : num_actions

                new_i = i + actions(action, 1);
                new_j = j + actions(action, 2);

                if ~(new_i >= 1 && new_i <= m && new_j >=1 && new_j <= n && R(new_i,new_j) ~= 1)
                    new_i = i;
                    new_j = j;
                end

                aux(action) = R(i, j) + gamma * V(new_i, new_j);
            end
            
            [V(i, j), pi(i,j)] = max(aux);
            delta = max(delta, abs(v - V(i, j)));            
        end
    end

    if delta < theta
        break;
    end
end

[~, policy] = max(pi, [], 3);

disp('Acciones: 1=arriba, 2=derecha, 3=abajo, 4=izquierda')
disp('Politica Optima')
disp(policy)

draw_heatmap(V)

start_position = [1 1];
[row, col] = find(R == 0);
draw_maze(R, start_position, pi, [row col])