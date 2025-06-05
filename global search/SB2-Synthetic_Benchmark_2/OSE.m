function traj_list = OSE(max_time, time_step, model, c, lb, ub, omega, CR, max_iter, level, select_dims, input_dims)

    steps = max_time / time_step;
    timestamps = (0:steps-1)' * time_step;
    init_inputs = zeros(steps, input_dims);
    
    p = [];
    disp(timestamps)
    disp(init_inputs)
    u = [timestamps, init_inputs];
    T = max_time;
    
    [ta, init_outputs] = feval(model, p, u, T);
    traj_list = {{init_inputs, init_outputs}};

    for iter = 1:max_iter

        disp(["Iteration: ", num2str(iter)]);
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
            d = dist(traj{2}, target_features);
            if d < min_dist
                min_dist = d;
                closest_traj = traj;
            end
        end

        closest_inputs = closest_traj{1};
        new_inputs = closest_inputs;

        delta = omega * tan(pi * (rand(size(closest_inputs)) - 0.5));
        mask = rand(size(closest_inputs)) < CR;
        new_inputs(mask) = closest_inputs(mask) + delta(mask);
        
        u = [timestamps, new_inputs];

        [ta, new_outputs] = feval(model, p, u, T);
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

    idx = sub2ind(size(traj), row_idx, col_idx);
    y2_vals = traj(idx);

    d = sum(((y2_vals - target) ./ scale) .^ 2);
end