classdef ExperienceReplay

    properties
        capacity
        memory = {}
        position = 1
    end

    methods
        function obj = ExperienceReplay(capacity)
            obj.capacity = capacity;
        end

        function obj = insert(obj, transition)
            if length(obj.memory) < obj.capacity
                obj.memory{end+1} = [];
            end

            obj.memory{obj.position} = transition;
            obj.position = mod(obj.position, obj.capacity) + 1;
        end

        function batch = sample(obj, batch_size)
            indices = randperm(length(obj.memory), batch_size);
            batch = obj.memory(indices);
        end

        function result = can_sample(obj, batch_size)
            result = length(obj.memory) >= batch_size;
        end
    end
end