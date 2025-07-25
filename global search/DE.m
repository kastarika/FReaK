classdef DE < handle
    properties
        max_time
        time_step
        simu_name
        population_size
        generations
        F % Differential weight
        CR % Crossover rate
        lb
        ub
        select_dims
        input_dims
        output_dims
        traj_list
        traj_idx
        stl_req
        cp
        falsified_file_name
        best_traj
        best_rob
    end
    
    methods
        function obj = DE(max_time, time_step, simu_name, population_size, generations, ...
                F, CR, input_dims, output_dims, ...
                select_dims, lb, ub, stl_req)
            obj.max_time = max_time;
            obj.time_step = time_step;
            obj.simu_name = simu_name;
            obj.population_size = population_size;
            obj.generations = generations;
            obj.F = F;
            obj.CR = CR;
            obj.input_dims = input_dims;
            obj.output_dims = output_dims;
            obj.select_dims = select_dims;
            obj.lb = lb;
            obj.ub = ub;
            obj.stl_req = stl_req;
            timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
            obj.falsified_file_name = [simu_name '_falsified_traj_' timestamp '.txt'];
            disp(obj.falsified_file_name);
            obj.best_rob = inf;
        end


        function run_DE(obj, sorted)
            if nargin < 2
                sorted = false;
            end

            obj.DE_producer();
            obj.traj_idx = 1;

            obj.visualize_de();

            if sorted
                obj.sort_traj();
            end
        end
        

        function new_traj = next_traj(obj, cp)
            output = obj.traj_list{obj.traj_idx}{2};
            first_row = output(1, :)';
            input = obj.traj_list{obj.traj_idx}{1};
            rest = input(1:cp,:);
            rest_flat = rest(:);
            new_traj = [first_row; rest_flat];

            obj.traj_idx = obj.traj_idx + 1;
            if obj.traj_idx > length(obj.traj_list)
                obj.run_DE(true);
            end
        end


        function sort_traj(obj)
            b_vars = arrayfun(@(i) sprintf('b_%d', i), 1:obj.output_dims, 'UniformOutput', false);
            Bdata = BreachTraceSystem(b_vars);

            for i = 1:length(obj.traj_list)
                traj = obj.traj_list{i}{2};
                time = 0:1:length(traj)-1;
                Bdata.AddTrace([time' traj]);
            end

            phi = STL_Formula('phi', obj.stl_req);
            robustness_vals = Bdata.GetRobustSat(phi);

            [~, idx] = sort(robustness_vals, 'descend'); % sort from most robust
            obj.traj_list = obj.traj_list(idx);

            figure;
            Bdata.PlotSignals();
            drawnow;

            Rphi = BreachRequirement(phi);
            Rphi.Eval(Bdata);
            BreachSamplesPlot(Rphi);
            drawnow;
        end
    end

    methods (Access=private)

        function DE_producer(obj)
            % Initialize population
            dim = obj.select_dims * obj.cp;
            disp(dim);
            obj.traj_list = cell(obj.population_size, 1);
            disp(obj.population_size);
            pop = rand(obj.population_size, dim) .* (obj.ub - obj.lb)' + obj.lb';
            obj.best_traj = rand(obj.cp * obj.select_dims) * (obj.ub - obj.lb) + obj.lb;

            % obj.visualize_de();
            for gen = 1:obj.generations
                for i = 1:obj.population_size
                    disp([gen, i]);
                    % Mutation
                    idxs = randperm(obj.population_size, 3);
                    while any(idxs == i)
                        idxs = randperm(obj.population_size, 3);
                    end
                    % x1 = pop(idxs(1), :);
                    x1 = obj.best_traj;
                    x2 = pop(idxs(2), :);
                    x3 = pop(idxs(3), :);
                    % disp(size(x1))
                    % disp(size(x2));
                    mutant = x1 + obj.F * (x2 - x3);

                    % Crossover
                    trial = pop(i, :);
                    jrand = randi(dim);
                    for j = 1:dim
                        if rand < obj.CR || j == jrand
                            trial(j) = mutant(j);
                        end
                    end
                    % trial = min(max(trial, -3), 4);
                    % Selection
                    f_trial = obj.evaluate_trajectory(trial);
                    f_target = obj.evaluate_trajectory(pop(i, :));
                    if f_trial < f_target
                        disp('yes');
                        pop(i, :) = trial;
                        if(f_trial < obj.best_rob)
                            obj.best_traj = trial;
                        end
                    end
                end
                
                
                for i = 1:obj.population_size
                    input = reshape(pop(i, :), [], obj.input_dims);
                    output = obj.simulate(input);
                    obj.traj_list{i} = {input, output};
                    % disp(i);
                    % celldisp(obj.traj_list{i});
                end
                obj.visualize_de();
                obj.write_falsified();
            end

            % Store trajectories
            % obj.traj_list = cell(obj.population_size, 1);
            % for i = 1:obj.population_size
            %     input = reshape(pop(i, :), [], obj.input_dims);
            %     output = obj.simulate(input);
            %     obj.traj_list{i} = {input, output};
            % end
        end


        function fitness = evaluate_trajectory(obj, params)
            input = reshape(params, [], obj.input_dims);
            output = obj.simulate(input);
            b_vars = arrayfun(@(i) sprintf('b_%d', i), 1:obj.output_dims, 'UniformOutput', false);
            Bdata = BreachTraceSystem(b_vars);
            time = 0:1:length(output)-1;
            % disp(time);
            % disp(output);
            Bdata.AddTrace([time' output]);
            phi = STL_Formula('phi', obj.stl_req);
            robustness = Bdata.GetRobustSat(phi);
            fitness = robustness; % Higher robustness = better
        end


        function output = simulate(obj, input)
            steps = obj.max_time / obj.time_step + 1;
            timestamps = (0:steps - 1)' * obj.time_step;
        
            p = []; % parameters (empty)
            % disp(input);
            % disp(timestamps);
            u = [timestamps, input];
        
            T = obj.max_time;
        
            % Call your simulator
            [ta, output] = runSimu(obj.simu_name, T, p, u);
        end


        function visualize_de(obj)
            b_vars = arrayfun(@(i) sprintf('b_%d', i), 1:obj.output_dims, 'UniformOutput', false);
            Bdata = BreachTraceSystem(b_vars);

            for i = 1:length(obj.traj_list)
                traj = obj.traj_list{i}{2};
                time = 0:1:length(traj)-1;
                Bdata.AddTrace([time' traj]);
            end

            phi = STL_Formula('phi', obj.stl_req);

            figure;
            Bdata.PlotSignals();
            drawnow;

            Rphi = BreachRequirement(phi);
            Rphi.Eval(Bdata);
            BreachSamplesPlot(Rphi);
            drawnow;
        end

        function write_falsified(obj)
            b_vars = arrayfun(@(i) sprintf('b_%d', i), 1:obj.output_dims, 'UniformOutput', false);
            Bdata = BreachTraceSystem(b_vars);

            for i = 1:length(obj.traj_list)
                traj = obj.traj_list{i}{2};
                time = 0:1:length(traj)-1;
                Bdata.AddTrace([time' traj]);
            end

            phi = STL_Formula('phi', obj.stl_req);
            robustness_vals = Bdata.GetRobustSat(phi);
            % Find indices where robustness < 0
            violating_idx = find(robustness_vals < 0);
            
            % Extract violating trajectories
            violating_trajs = obj.traj_list(violating_idx);
            
            
            % Prepare file to append
            fileID = fopen(obj.falsified_file_name, 'a'); % 'a' = append mode
            
            % Write simulation name at the top (once per run)
            fprintf(fileID, 'Simulation: %s\n', obj.simu_name);
            fprintf(fileID, '=============================\n');

            % Loop through and write each violating trajectory
            for k = 1:length(violating_trajs)
                traj = violating_trajs{k};
                input = traj{1};
                if any(input > 1 | input < 0)
                    continue
                end
                output = traj{2};
                
                fprintf(fileID, 'Trajectory #%d\n', violating_idx(k));
                fprintf(fileID, 'Input:\n');
                dlmwrite(obj.falsified_file_name, input, '-append', 'delimiter', '\t');
                fprintf(fileID, 'Output:\n');
                dlmwrite(obj.falsified_file_name, output, '-append', 'delimiter', '\t');
                fprintf(fileID, '-----------------------------\n');

            end
            
            % Close the file
            fclose(fileID);

        end

    end
end
