
% Simulation settings
max_time = 24;
time_step = 6;
input_dims = 1;
steps = max_time / time_step;

% Bounds
lb = zeros(steps, input_dims);
ub = ones(steps, input_dims);
ub(:) = 100;

% OSE parameters
c = 0.1;
omega = 0.2;
CR = 0.8;
max_iter = 100;
level = 2;
select_dims = [1];
model = 'run_synth_benchmark2';  

% Run OSE
traj_list = OSE(max_time, time_step, model, c, lb, ub, omega, CR, max_iter, level, select_dims, input_dims);


Bdata = BreachTraceSystem({'b'});

for i = 1:length(traj_list)
    traj = traj_list{i}{2};
    time = 0:1:length(traj)-1;
    %disp(length(time));
    %disp(length(traj));
    % Process trajectory data for Breach
    Bdata.AddTrace([time' traj]);
end
figure;Bdata.PlotSignals();

phi = STL_Formula('phi', 'alw_[0,18]( b[t]>90 or ev_[0,6](b[t]<50) )');
Rphi = BreachRequirement(phi);

disp(Rphi.Eval(Bdata));
BreachSamplesPlot(Rphi);
figure;Rphi.PlotRobustSat();

