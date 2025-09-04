function [rob, grad] = robustness_gradient_fd(phi_str, X)
    % robustness_gradient_fd
    % Numerically compute robustness and gradient wrt states using finite differences
    %
    % phi_str : STL formula as a string
    % X       : struct of trajectories (each field is a state, values are row vectors)

    var_names = fieldnames(X);   % state/signal names
    T = length(X.(var_names{1}));  % trajectory length (assume all same length)

    % Build baseline trace
    sig_cells = cellfun(@(f) X.(f)(:), var_names, 'UniformOutput', false);
    signal_mat = cat(2, sig_cells{:});
    time = (0:T-1)';   % assume unit step
    trace_mat = [time signal_mat];

    % Create Breach trace system
    Bdata = BreachTraceSystem(var_names);
    Bdata.AddTrace(trace_mat);

    % STL formula
    phi = STL_Formula('phi', phi_str);

    % Compute baseline robustness
    rob = Bdata.GetRobustSat(phi);

    % Finite difference step
    eps = 1e-6;
    grad = struct();
    for i = 1:length(var_names)
        grad.(var_names{i}) = zeros(1, T);
    end

    % Compute numerical gradient
    for i = 1:length(var_names)
        for t = 1:T
            % Perturb forward
            X_pert = X;
            X_pert.(var_names{i})(t) = X_pert.(var_names{i})(t) + eps;

            % Build perturbed trace
            sig_cells = cellfun(@(f) X_pert.(f)(:), var_names, 'UniformOutput', false);
            signal_mat = cat(2, sig_cells{:});
            trace_mat = [time signal_mat];

            Bdata_pert = BreachTraceSystem(var_names);
            Bdata_pert.AddTrace(trace_mat);

            rob_pert = Bdata_pert.GetRobustSat(phi);

            % Finite difference
            grad.(var_names{i})(t) = (rob_pert - rob) / eps;
        end
    end
end
