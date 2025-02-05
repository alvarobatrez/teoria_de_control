close all; clear, clc

M = create_maze();
[goal_row, goal_col] = find(M==0);
actions = [-1 0; 0 1; 1 0; 0 -1]; % arriba, derecha, abajo, izquierda

[m, n] = size(M);
num_actions = length(actions);

pi = randi(num_actions, m, n);
pi(M==-2) = 0;
pi(M==10) = 0;
V = zeros(m, n);

theta = 1e-6;
gamma = 0.99;

while true
    V = policy_evaluation(M, pi, V, theta, gamma, actions, m ,n);
    [V, pi, policy_stable] = policy_improvement(M, pi, V, gamma, actions, num_actions, m, n);

    if policy_stable == true
        break
    end
end

disp('Acciones: 1=arriba, 2=derecha, 3=abajo, 4=izquierda')
disp('Politica Optima')
disp(pi)

draw_heatmap(V)

start_position = [1 2];
draw_maze(M, start_position, pi, [goal_row goal_col])

function V = policy_evaluation(M, pi, V, theta, gamma, actions, m, n)
    while true
        delta = 0;

        for i = 1 : m
            for j = 1 : n
                if M(i, j) == -2 || M(i, j) == 0
                    continue
                end
                
                v = V(i, j);
                action = pi(i, j);
                new_i = i + actions(action, 1);
                new_j = j + actions(action, 2);

                if new_i < 1 || new_i > m || new_j < 1 || new_j > n || M(new_i, new_j) == -2
                    new_i = i;
                    new_j = j;
                end

                V(i, j) = M(new_i, new_j) + gamma * V(new_i, new_j);
                delta = max(delta, abs(v - V(i,j)));
            end
        end

        if delta < theta
            break;
        end
    end
end

function [V, pi, policy_stable] = policy_improvement(M, pi, V, gamma, actions, num_actions, m, n)
    policy_stable = true;

    for i = 1 : m
        for j = 1 : n
            if M(i, j) == -2 || M(i, j) == 0
                continue
            end

            old_action = pi(i, j);
            aux = zeros(1, num_actions);
            
            for action = 1 : num_actions
                new_i = i + actions(action, 1);
                new_j = j + actions(action, 2);

                if new_i < 1 || new_i > m || new_j < 1 || new_j > n || M(new_i, new_j) == -2
                    new_i = i;
                    new_j = j;
                end

                aux(action) = M(new_i, new_j) + gamma * V(new_i, new_j);
            end

            [~, pi(i, j)] = max(aux);

            if old_action ~= pi(i, j)
                policy_stable = false;
            end
        end
    end
end

function draw_heatmap(V)
    h = heatmap(V);
    h.ColorbarVisible = 'off';
    colormap(h, 'hot')
    title('Función de Valor')
end