ose = t2();
p=[];
steps = ose.max_time / ose.time_step + 1;
timestamps = (0:steps - 1)' * ose.time_step;

b_vars = arrayfun(@(i) sprintf('b_%d', i), 1:ose.output_dims, 'UniformOutput', false);
Bdata = BreachTraceSystem(b_vars);
for i = 1:length(ose.traj_list)
    traj = ose.traj_list{i}{2};
    time = 0:1:length(traj)-1;
    Bdata.AddTrace([time' traj]);
end
phi = STL_Formula('phi', ose.stl_req);
robustness_vals = Bdata.GetRobustSat(phi);
% disp(robustness_vals);
[~, idx] = sort(robustness_vals);
% % disp('best from sort')
% celldisp(ose.traj_list(idx(1)));
ose.traj_list = ose.traj_list(idx);


pert_traj = [];
n = length(ose.traj_list{1}(1));

CR = 0.8;

simu_name = 'phi2_m1_vr01_k2_2';

for k = 1:10
    newBdata = BreachTraceSystem(b_vars);
    for i = 1:40
        for j = 1:4
            disp([i, j]);
            jj = ose.traj_list{i};
            closest_inputs = jj{1};
            new_inputs = closest_inputs;
            % disp(new_inputs);
            omega = 0.1;
            delta = omega * tan(pi * (rand(size(closest_inputs)) - 0.5));
            mask = rand(size(closest_inputs)) < CR;
            % disp(delta);
            % disp(mask);
            % disp(delta(mask));
            new_inputs(mask) = closest_inputs(mask) + delta(mask);
            new_inputs = min(max(new_inputs, 0), 1);
            
    
            u = [timestamps, new_inputs];
    
            [ta, new_outputs] = runSimu(simu_name, T, p, u);
    
            traj = new_outputs;
            time = 0:1:length(traj)-1;
            newBdata.AddTrace([time' traj]);
            pert_traj{end+1} = {new_inputs, new_outputs};
        end
    end
    
    figure;
    newBdata.PlotSignals();
    
    Rphi = BreachRequirement(phi);
    Rphi.Eval(newBdata);
    BreachSamplesPlot(Rphi);
    drawnow();
    pause(0.1);
    ose.traj_list = pert_traj;
    ose.sort_traj();
    
    disp('end');
end