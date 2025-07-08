classdef OSE < handle
    properties
        max_time
        time_step
        simu_name
        c
        ro
        lb
        ub
        omega
        CR
        max_iter
        select_dims
        input_dims
        output_dims
        traj_list
        traj_idx
        stl_req
        cp
    end
    methods
        function obj = OSE(max_time, time_step, simu_name, c, ro, lb, ub, omega, CR, max_iter, select_dims, input_dims, stl_req)
            % exist to easily remind what properties are needed
        end

        function run_OSE_input(obj)
            sorted = true;

            obj.OSE_input_producer();
            obj.traj_idx = 1; % Initialize trajectory index

            if(sorted)
                obj.sort_traj(); % Sort trajectories if the sorted flag is true
            end
        end


        function run_OSE(obj, sorted)
            if(nargin < 2)
                sorted = false; % Default to false if not provided
            end
            
            sorted = true;

            obj.OSE_producer();
            obj.traj_idx = 1; % Initialize trajectory index

            if(sorted)
                obj.sort_traj(); % Sort trajectories if the sorted flag is true
            end
        end
        
        function new_traj = next_traj(obj)
            output = obj.traj_list{obj.traj_idx}{2};
            first_row = output(1, :)';
            % disp(output);
            % disp(first_row);
            input = obj.traj_list{obj.traj_idx}{1};
            % rest = input(2:end,:);
            rest = input(1:end,:);
            rest_flat = rest(:);
            new_traj = [first_row; rest_flat];

            obj.traj_idx = obj.traj_idx + 1; % Update trajectory index for next call
            % if obj.traj_idx > length(obj.traj_list)
            if obj.traj_idx > 60
                obj.run_OSE(true);
            end
        end
        % disp(new_traj);
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
            % disp(robustness_vals);
            [~, idx] = sort(robustness_vals);
            % disp('best from sort')
            % celldisp(obj.traj_list(idx(1)));
            obj.traj_list = obj.traj_list(idx);
            % disp('best after sort');
            % disp(idx);
            % celldisp(obj.traj_list(1));
            

            figure;
            Bdata.PlotSignals();
            drawnow;
            pause(0.1);
            Rphi = BreachRequirement(phi);
            Rphi.Eval(Bdata);
            BreachSamplesPlot(Rphi);
            drawnow;
            pause(0.1);
        end
    end

    methods(Access=private)

        function OSE_producer(obj)
            obj.traj_list = OSE_output(obj.max_time, obj.time_step, obj.simu_name, ...
                obj.c, obj.ro, obj.lb, obj.ub, obj.omega, obj.CR, ...
                obj.max_iter, obj.select_dims, obj.input_dims);
            disp(size(obj.traj_list));
        end
        function OSE_input_producer(obj)
            obj.traj_list = OSE_input(obj.max_time, obj.time_step, obj.simu_name, ...
                obj.c, obj.ro, obj.lb, obj.ub, obj.omega, obj.CR, ...
                obj.max_iter, obj.select_dims, obj.input_dims);
            disp(size(obj.traj_list));
        end
        
        
    end
end
