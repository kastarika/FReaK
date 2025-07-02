% the OSE producer produces a trajectory list filled in using OSE
% it takes in an OSE object with parameters needed for OSE
% it also takes in parameters regarding the graphs produced by the OSE
% (used for diagnostics)
function sorted_traj = OSE_producer(obj, input_plot, output_plot, robust_plot)
    
    if(nargin < 4)
        robust_plot = 0;
    end
    if(nargin < 3)
        output_plot = 0;
    end
    if(nargin < 2)
        input_plot = 0;
    end


    % Simulation settings
    max_time = obj.max_time;
    time_step = obj.time_step;
    input_dims = obj.input_dims;
    
    % Bounds
    % lower bound and upper bound for observed output
    lb = obj.lb;
    ub = obj.ub;


    
    % OSE parameters
    c = obj.c; % a coefficient determining how much outside the observed output we want to explore in feature selection 
    omega = obj.omega; % the mean for cauchy distribution
    CR = obj.CR; % probability for evolution of (changing) a single point in input when producing new input
    max_iter = obj.max_iter;
    select_dims = obj.select_dims; % the output dimensions used for feature selection
    model = obj.model;  % name of the matlab function for running the model
    ro = obj.ro; % the probability for choosing a level (how many output points we want to include in feature selection)
    
    % Run OSE
    traj_list = OSE(max_time, time_step, model, c, ro, lb, ub, omega, CR, max_iter, select_dims, input_dims);
    
    
    time = (0:4)';
    
    if input_plot
        figure;
        hold on;
        for i = 1:length(traj_list) 
            traj = traj_list{i}{1};
            plot3(time, traj(:, 1), traj(:, 2));
        end
        xlabel('Time')
        ylabel('Input Dim 1')
        zlabel('Input Dim 2')
        title('3D Plot of Trajectories')
        grid on;
        view(3);
    end
    
    Bdata = BreachTraceSystem({'b_1', 'b_2'});
    
    for i = 1:length(traj_list)
        traj = traj_list{i}{2};
        time = 0:1:length(traj)-1;
        Bdata.AddTrace([time' traj]);
    end
    
    if(output_plot)
        figure;
        Bdata.PlotSignals();
    end
    
    phi = STL_Formula('phi', 'alw_[0,19]((alw_[0,5](b_1[t] <= 20) or (ev_[0,5](b_2[t] >= 40))))');
    Rphi = BreachRequirement(phi);
    
    disp(Rphi.Eval(Bdata));
    if(robust_plot)
        BreachSamplesPlot(Rphi);
    end
    % figure;Rphi.PlotRobustSat();
    
    % sort based on robustness and return the trajectories
    robustness_vals = Bdata.GetRobustSat(phi);
    [~, idx] = sort(robustness_vals);
    sorted_traj = traj_list(idx);

end



