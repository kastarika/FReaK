input_dims = 1;
output_dims = 1;
max_time = 24;
time_step = 6;
cp = max_time/time_step + 1;
num_t = 10;

b_vars = arrayfun(@(i) sprintf('b_%d', i), 1:output_dims, 'UniformOutput', false);
Bdata = BreachTraceSystem(b_vars);
wholedata = BreachTraceSystem(b_vars);

input = zeros(cp, input_dims);
input(1) = 0.35;
input(2) = 0;
disp(input);
eflag = false;
cnt = 0;
cnt2 = 0;
tmp_trajes = [];
best_trajes = [];
while cnt2 < 15
    cnt = cnt + 1;
    [input, eflag] = inc_input(3, input, num_t, cp);
    steps = max_time / time_step + 1;
    timestamps = (0:steps - 1)' * time_step;
    u = [timestamps, input];
    p = [];
    T = max_time;
    [tout, output] = run_synth_benchmark2(p, u, T);
    tmp_trajes{end + 1} = {input, output};
    disp(input);
    traj = output;
    time = 0:1:length(traj)-1;
    Bdata.AddTrace([time' traj]);
    if cnt > 100
        cnt = 0;
        cnt2 = cnt2 + 1;
        phi = STL_Formula('phi', 'alw_[0,18] (b_1[t] > 90 or (ev_[0,6] (b_1[t] < 50)))');
        figure;
        Bdata.PlotSignals();
        Rphi = BreachRequirement(phi);
        Rphi.Eval(Bdata);
        BreachSamplesPlot(Rphi);
        
        drawnow;
        pause(0.1);
        robustness_vals = Bdata.GetRobustSat(phi);
        idxx = find(robustness_vals < 7);
        % disp(idxx)
        trajes = Bdata.GetTraces();
        % celldisp(trajes);
        for iii = 1:length(idxx)
            wholedata.AddTrace(trajes{idxx(iii)});
            best_trajes{end + 1} = tmp_trajes{idxx(iii)};
        end
        Bdata = BreachTraceSystem(b_vars);
        % break;
    end

end

phi = STL_Formula('phi', 'alw_[0,18] (b_1[t] > 90 or (ev_[0,6] (b_1[t] < 50)))');
figure;
wholedata.PlotSignals();
Rphi = BreachRequirement(phi);
Rphi.Eval(wholedata);
BreachSamplesPlot(Rphi);
save('best_trajes.mat', 'best_trajes');
% disp(class(best_trajes));
% celldisp(best_trajes);
writecell(flattenCellForWrite(best_trajes), 'best_trajes.txt');


function [new_input, eflag] = inc_input(t, input, num_t, cp)
    new_input = input; 
    if t > cp
        eflag = true;
        return;
    else
        eflag = false;
    end
       
    new_input(t) = new_input(t) + 1/num_t;
    if new_input(t) > 1
        new_input(t) = 0;
        [new_input, eflag] = inc_input(t + 1, new_input, num_t, cp);
    end

end

function C_flat = flattenCellForWrite(C)
    % Recursively flatten cell array elements for writing to file
    C_flat = C;
    for i = 1:numel(C_flat)
        C_flat{i} = flattenElement(C_flat{i});
    end
end

function out = flattenElement(x)
    if iscell(x)
        % Recursively flatten inner cells and join with commas
        flattened = cellfun(@flattenElement, x, 'UniformOutput', false);
        out = strjoin(string(flattened), ', ');
    elseif isstruct(x)
        % Convert struct to JSON string
        out = jsonencode(x);
    elseif isnumeric(x) || islogical(x)
        if isscalar(x)
            out = x; % leave scalars as-is
        else
            % Convert array to string like [1 2 3]
            out = mat2str(x);
        end
    elseif ischar(x) || isstring(x)
        out = string(x);
    else
        % For unsupported types, use class name
        out = sprintf('<%s>', class(x));
    end
end

