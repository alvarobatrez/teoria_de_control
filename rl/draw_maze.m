function draw_maze(maze, agent_position, policy, exit)

    actions = [-1 0; 0 1; 1 0; 0 -1];
    maze(maze == -1) = 2;
    figure

    imagesc(maze)
    colormap([1 0 0; 0 0 0; 1 1 1])
    hold on
    axis off
    axis equal

    agent_marker = plot(agent_position(2), agent_position(1), 'bo', 'MarkerSize', 30, 'MarkerFaceColor', 'b');
    title('Maze')

    while true
        pause(0.5)

        if isequal(agent_position, exit)
            break
        end

        policy_selected = policy(agent_position(1), agent_position(2));
        agent_position = agent_position + actions(policy_selected, :);
        
        set(agent_marker, 'XData', agent_position(2), 'YData', agent_position(1));
        drawnow
    end

    hold off
end