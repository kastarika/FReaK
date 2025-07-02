function traj_list = OSE_input(max_time, time_step, simu_name, c, ro, lb, ub, omega, CR, max_iter, select_dims, input_dims)

    steps = max_time / time_step + 1;
    timestamps = (0:steps - 1)' * time_step;
    init_inputs = ones(steps, input_dims) * 0.5;
    
    p = [];
    % disp(timestamps);
    % disp(init_inputs);
    u = [timestamps, init_inputs];
    T = max_time;
    
    [ta, init_outputs] = runSimu(simu_name, T, p, u);
    traj_list = {{init_inputs, init_outputs}};

    for iter = 1:max_iter

        disp(["Iteration: ", num2str(iter)]);
        level = randsample(length(ro), 1, true, ro);
        target_features = cell(1, level);

        for j = 1:level
            feature = zeros(length(select_dims), 4);
            for k = 1:length(select_dims)
                i = select_dims(k);
                t_idx = randi([1, steps]);
                dif = ub(t_idx, i) - lb(t_idx, i);
                new_f = (1 + 2 * c) * dif * rand() + lb(t_idx, i) - c * dif;
                feature(k, :) = [new_f, t_idx, i, dif];
            end
            target_features{j} = feature;
        end

        % Find the closest trajectory
        min_dist = Inf;
        closest_traj = {};
        for idx = 1:length(traj_list)
            traj = traj_list{idx};
            d = dist(traj{1}, target_features);
            if d < min_dist
                min_dist = d;
                closest_traj = traj;
            end
        end
        % disp(target_features);
        closest_inputs = closest_traj{1};
        % disp(closest_inputs);
        new_inputs = closest_inputs;
        % disp(new_inputs);
        delta = omega * tan(pi * (rand(size(closest_inputs)) - 0.5));
        %delta = omega * tan(pi * (rand(size(closest_inputs))));
        mask = rand(size(closest_inputs)) < CR;
        new_inputs(mask) = closest_inputs(mask) + delta(mask);
        new_inputs = min(max(new_inputs, 0), 1);

        u = [timestamps, new_inputs];

        [ta, new_outputs] = runSimu(simu_name, T, p, u);
        %disp(new_outputs)
        traj_list{end+1} = {new_inputs, new_outputs};
    end
end

function d = dist(traj, features)
    % Flatten multi-dimensional cell array of features
    if iscell(features)
        features = cat(1, features{:});
    end

    target = features(:, 1);
    row_idx = features(:, 2);
    col_idx = features(:, 3);
    scale = features(:, 4);
        
    % disp('adsfasdf');
    % disp(traj);
    % disp('adsfasdf');
    % disp(size(traj));
    % disp('adsfasdf');
    % disp(row_idx);
    % disp('adsfasdf');
    % disp(col_idx);
    % disp('adsfasdf');


    idx = sub2ind(size(traj), row_idx, col_idx);
    y2_vals = traj(idx);

    d = sum(((y2_vals - target) ./ scale) .^ 2);
end