
% Simulation settings
max_time = 24;
time_step = 6;
input_dims = 2;
steps = max_time / time_step;

% Bounds
lb = zeros(max_time + 1, input_dims);
ub = ones(max_time + 1, input_dims);
lb(:) = -30;
ub(:) = 30;

% OSE parameters
c = 0.2;
omega = 0.15;
CR = 0.6;
max_iter = 500;
level = 2;
select_dims = [1];
model = 'run_synth_benchmark4';  
ro = [4/7, 2/7, 1/7];

% Run OSE
traj_list = OSE(max_time, time_step, model, c, ro, lb, ub, omega, CR, max_iter, select_dims, input_dims);

disp(size(traj_list))

%[T, D, N] = size(traj_list);  % T = timesteps, D = input dimensions, N = trajectories
time = (0:4)';

figure;
hold on;
for i = 1:length(traj_list)
    traj = traj_list{i}{1};  % size T×
    %disp(traj);
    %disp(traj(:, 1));
    plot3(time, traj(:, 1), traj(:, 2));
end
xlabel('Time')
ylabel('Input Dim 1')
zlabel('Input Dim 2')
title('3D Plot of Trajectories')
grid on;
view(3);


Bdata = BreachTraceSystem({'b_1', 'b_2'});

for i = 1:length(traj_list)
    traj = traj_list{i}{2};
    time = 0:1:length(traj)-1;
    %disp(length(time));
    %disp(length(traj));
    % Process trajectory data for Breach
    Bdata.AddTrace([time' traj]);
end

figure;Bdata.PlotSignals();

phi = STL_Formula('phi', 'alw_[0,19]((alw_[0,5](b_1[t] <= 20) or (ev_[0,5](b_2[t] >= 40))))');
Rphi = BreachRequirement(phi);

disp(Rphi.Eval(Bdata));
BreachSamplesPlot(Rphi);
figure;Rphi.PlotRobustSat();

