function draw_heatmap(V)
h = heatmap(V);
h.ColorbarVisible = 'off';
colormap(h, 'hot')
title('Value Function')