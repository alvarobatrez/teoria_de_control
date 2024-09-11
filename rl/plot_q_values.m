function plot_q_values(Q)
[rows, cols, actions] = size(Q);

figure, hold on
set(gcf, 'Position', [100, 100, 700, 700]);
for i = 1 : rows
    for j = 1 : cols
        for a = 1 : actions
            value = Q(i,j,a);
            if a == 1
                x = [j j+0.5 j+1];
                y = [i i+0.5 i];
                text_pos = [j+0.5 i+0.25];
            elseif a == 2
                x = [j+0.5 j+1 j+1];
                y = [i+0.5 i+1 i];
                text_pos = [j+0.75 i+0.5];
            elseif a == 3
                x = [j j+1 j+0.5];
                y = [i+1 i+1 i+0.5];
                text_pos = [j+0.5 i+0.75];
            else
                x = [j j j+0.5];
                y = [i i+1 i+0.5];
                text_pos = [j+0.25 i+0.5];
            end
            patch(x, rows-y, value, 'EdgeColor','none');
            text(text_pos(1), rows-text_pos(2), sprintf('%.1f', value), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 7)
        end
    end
end
colormap('hot')
title('Action-Value Function')