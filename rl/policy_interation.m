clear, clc

R = [0 -10 -10 -10; -1 -1 -1 -1; -10 -10 -10 -1; -1 -1 -1 -1];
actions = [-1 0; 0 1; 1 0; 0 -1];

[m, n] = size(R);
num_actions = length(actions);

pi = randi(num_actions, m, n);
pi(R==0) = 0;
theta = 0.001;
gamma = 0.9;

while true

    V = policy_evaluation(R, pi, theta, gamma, actions, m, n);

    [V, pi, policy_stable] = policy_improvement(R, pi, V, gamma, actions, num_actions, m, n);

    if policy_stable == true
        break
    end
end

disp('Matriz de Recompensas')
disp(R)

disp('V(s)')
disp(V)

disp('Acciones: 1=arriba, 2=derecha, 3=abajo, 4=izquierda')

disp('Politica')
disp(pi)


function V = policy_evaluation(R, pi, theta, gamma, actions, m, n)
    V = zeros(m, n);
    
    while true
        delta = 0;
    
        for i = 1 : m
            for j = 1 : n
    
                if R(i, j) == 0
                    continue
                end
    
                v = V(i,j);
                action = pi(i, j);
                new_i = i + actions(action, 1);
                new_j = j + actions(action, 2);
    
                if ~(new_i >= 1 && new_i <= m && new_j >=1 && new_j <= n)
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
            old_action = pi(i, j);
            aux = zeros(1, num_actions);

            for action = 1 : num_actions

                new_i = i + actions(action, 1);
                new_j = j + actions(action, 2);

                if ~(new_i >= 1 && new_i <= m && new_j >=1 && new_j <= n)
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