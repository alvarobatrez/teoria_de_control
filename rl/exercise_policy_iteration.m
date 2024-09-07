close all; clear, clc

R = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

[m, n] = size(R);
num_actions = length(actions);

pi = randi(num_actions, m, n);
pi(R==0) = 0;
pi(R==1) = 0;
V = zeros(m, n);

theta = 1e-6;
gamma = 0.9;

while true

    V = policy_evaluation(R, pi, V, theta, gamma, actions, m, n);

    [V, pi, policy_stable] = policy_improvement(R, pi, V, gamma, actions, num_actions, m, n);

    if policy_stable == true
        break
    end
end

disp('Acciones: 1=arriba, 2=derecha, 3=abajo, 4=izquierda')
disp('Politica Optima')
disp(pi)

draw_heatmap(V)

start_position = [1 1];
[row, col] = find(R == 0);
draw_maze(R, start_position, pi, [row col])

function V = policy_evaluation(R, pi, V, theta, gamma, actions, m, n)
    
    while true
        delta = 0;
    
        for i = 1 : m
            for j = 1 : n
    
                if R(i, j) == 0 || R(i, j) == 1
                    continue
                end
    
                v = V(i,j);
                action = pi(i, j);
                new_i = i + actions(action, 1);
                new_j = j + actions(action, 2);
    
                if ~(new_i >= 1 && new_i <= m && new_j >=1 && new_j <= n && R(new_i,new_j) ~= 1)
                    new_i = i;
                    new_j = j;
                end
    
                V(i, j) = R(i, j) + gamma * V(new_i, new_j);
                delta = max(delta, abs(v - V(i, j)));
            end
        end
    
        if delta < theta
            break;
        end
    end
end

function [V, pi, policy_stable] = policy_improvement(R, pi, V, gamma, actions, num_actions, m, n)
    policy_stable = true;

    for i = 1 : m
        for j = 1 : n

            if R(i, j) == 0 || R(i, j) == 1
                continue
            end
            
            old_action = pi(i, j);
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
            [~, pi(i,j)] = max(aux);

            if old_action ~= pi(i,j)
                policy_stable = false;
            end
        end
    end
end