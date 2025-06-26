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
        traj_list
        traj_idx
        stl_req
    end
    methods
        function obj = OSE(max_time, time_step, simu_name, c, ro, lb, ub, omega, CR, max_iter, select_dims, input_dims, stl_req)
            % exist to easily remind what properties are needed
        end

        function run_OSE(obj, sorted)
            if(nargin < 2)
                sorted = false; % Default to false if not provided
            end

            obj.OSE_producer();
            obj.traj_idx = 1; % Initialize trajectory index

            if(sorted)
                obj.sort_traj(); % Sort trajectories if the sorted flag is true
            end
        end
        
        function new_traj = next_traj(obj)
            new_traj = obj.traj_list{obj.traj_idx}; % Retrieve the next trajectory
            obj.traj_idx = obj.traj_idx + 1; % Update trajectory index for next call
            if obj.traj_idx > length(obj.traj_list)
                obj.run_OSE();
            end
        end

    end

    methods(Access=private)
        
        function OSE_producer(obj)
            obj.traj_list = OSE_output(obj.max_time, obj.time_step, obj.simu_name, ...
                obj.c, obj.ro, obj.lb, obj.ub, obj.omega, obj.CR, ...
                obj.max_iter, obj.select_dims, obj.input_dims);
        end
        
        function sort_traj(obj)
            output_dims = size(obj.traj_list{1}{2}, 2);
            b_vars = arrayfun(@(i) sprintf('b_%d', i), 1:output_dims, 'UniformOutput', false);
            Bdata = BreachTraceSystem(b_vars);
            
            for i = 1:length(obj.traj_list)
                traj = obj.traj_list{i}{2};
                time = 0:1:length(traj)-1;
                Bdata.AddTrace([time' traj]);
            end
            phi = STL_Formula('phi', obj.stl_req);
            robustness_vals = Bdata.GetRobustSat(phi);
            [~, idx] = sort(robustness_vals);
            obj.traj_list = obj.traj_list(idx);
        end
    end
end
