function [u_opt, fval, exitflag, info] = koopman_hybrid(A, B, g, x0, phi_str, u0, lb, ub, N, options)
    % Hybrid optimization using both gradient descent and simulated annealing
    % for Koopman model falsifying trajectory search
    
    if nargin < 10
        options = struct();
    end
    
    % Default options
    default_opts = struct(...
        'use_sa_first', false, ...           % Start with SA then GD
        'sa_max_iter', 10, ...             % SA iterations
        'sa_initial_temp', 1, ...         % Initial temperature
        'sa_final_temp', 0.001, ...          % Final temperature
        'sa_cooling_rate', 0.8, ...        % Cooling schedule
        'gd_max_iter', 50, ...             % GD iterations
        'alternating_cycles', 3, ...        % Number of SA-GD cycles
        'early_stop_threshold', -inf, ... % Stop when robustness < threshold
        'verbose', true ...                 % Display progress
    );
    
    % Merge user options with defaults
    opts = merge_options(default_opts, options);
    
    % Dimensions
    dim_x = length(x0);
    dim_u = size(B,2);
    x0 = x0(:);
    z0 = g(x0);
    dim_z = length(z0);
    
    % Projection matrix
    C = [eye(dim_x), zeros(dim_x, dim_z - dim_x)];
    
    % Precompute A^(k-1)*B
    AB_cells = cell(N,1);
    A_power = eye(dim_z);
    for k = 1:N
        AB_cells{k} = A_power * B;
        A_power = A * A_power;
    end
    
    % Bounds
    lb_vec = repmat(lb, N, 1);
    ub_vec = repmat(ub, N, 1);
    
    % Objective function wrapper
    fun = @(u) stl_cost_devec_pre(u, A, B, g, z0, phi_str, C, dim_u, N, AB_cells);
    
    % Initialize results
    u_current = u0(:);
    [fval_current, ~] = fun(u_current);
    
    % Best solution tracking
    u_best = u_current;
    fval_best = fval_current;
    
    % Results tracking
    info = struct();
    info.history = [];
    info.method_sequence = {};
    info.sa_accepts = 0;
    info.sa_rejects = 0;
    
    if opts.verbose
        fprintf('Starting hybrid optimization (SA + GD)\n');
        fprintf('Initial robustness: %.6f\n', fval_current);
    end
    
    %% Main optimization loop
    for cycle = 1:opts.alternating_cycles
        
        %% Simulated Annealing Phase
        if opts.verbose
            fprintf('\n--- Cycle %d: Simulated Annealing Phase ---\n', cycle);
        end
        
        [u_current, fval_current, sa_info] = simulated_annealing_phase(...
            u_current, fun, lb_vec, ub_vec, opts);
        
        % Update best solution
        if fval_current < fval_best
            u_best = u_current;
            fval_best = fval_current;
        end
        
        % Track results
        info.sa_accepts = info.sa_accepts + sa_info.accepts;
        info.sa_rejects = info.sa_rejects + sa_info.rejects;
        info.method_sequence{end+1} = 'SA';
        
        if opts.verbose
            fprintf('SA Phase completed. Current robustness: %.6f, Best: %.6f\n', ...
                fval_current, fval_best);
        end
        
        % Early stopping check
        if fval_best < opts.early_stop_threshold
            if opts.verbose
                fprintf('Early stopping: robustness threshold reached\n');
            end
            break;
        end
        
        %% Gradient Descent Phase
        if opts.verbose
            fprintf('\n--- Cycle %d: Gradient Descent Phase ---\n', cycle);
        end
        
        [u_current, fval_current, gd_exitflag] = gradient_descent_phase(...
            u_current, fun, lb_vec, ub_vec, opts);
        
        % Update best solution
        if fval_current < fval_best
            u_best = u_current;
            fval_best = fval_current;
        end
        
        info.method_sequence{end+1} = 'GD';
        
        if opts.verbose
            fprintf('GD Phase completed. Current robustness: %.6f, Best: %.6f\n', ...
                fval_current, fval_best);
        end
        
        % Early stopping check
        if fval_best < opts.early_stop_threshold
            if opts.verbose
                fprintf('Early stopping: robustness threshold reached\n');
            end
            break;
        end
    end
    
    % Final results
    u_opt = u_best;
    fval = fval_best;
    exitflag = (fval < opts.early_stop_threshold);
    
    info.final_robustness = fval;
    info.cycles_completed = cycle;
    
    if opts.verbose
        fprintf('\nOptimization completed!\n');
        fprintf('Final robustness: %.6f\n', fval);
        fprintf('SA acceptance rate: %.2f%%\n', ...
            100 * info.sa_accepts / (info.sa_accepts + info.sa_rejects));
    end
end

%% Simulated Annealing Phase
function [u_best, fval_best, info] = simulated_annealing_phase(u0, fun, lb, ub, opts)
    
    u_current = u0;
    [fval_current, ~] = fun(u_current);
    
    u_best = u_current;
    fval_best = fval_current;
    
    % SA parameters
    T = opts.sa_initial_temp;
    T_final = opts.sa_final_temp;
    cooling_rate = opts.sa_cooling_rate;
    
    accepts = 0;
    rejects = 0;
    
    for iter = 1:opts.sa_max_iter
        disp(iter)
        % Generate candidate solution with adaptive step size
        step_size = T / opts.sa_initial_temp; % Normalize by initial temp
        u_candidate = generate_candidate(u_current, lb, ub, step_size);
        
        % Evaluate candidate
        [fval_candidate, ~] = fun(u_candidate);
        
        % Acceptance criterion (minimization: accept if candidate is better or with probability)
        delta = fval_candidate - fval_current;
        
        if delta <= 0 || rand() < exp(-delta/T)
            % Accept candidate
            u_current = u_candidate;
            fval_current = fval_candidate;
            accepts = accepts + 1;
            
            % Update best solution
            if fval_candidate < fval_best
                u_best = u_candidate;
                fval_best = fval_candidate;
            end
        else
            rejects = rejects + 1;
        end
        
        % Cool down
        T = max(T * cooling_rate, T_final);
        
        % Progress display
        if opts.verbose && mod(iter, 100) == 0
            fprintf('SA iter %d: T=%.4f, current=%.6f, best=%.6f\n', ...
                iter, T, fval_current, fval_best);
        end
    end
    
    info = struct();
    info.accepts = accepts;
    info.rejects = rejects;
end

%% Gradient Descent Phase
function [u_opt, fval, exitflag] = gradient_descent_phase(u0, fun, lb, ub, opts)
    
    % fmincon options for gradient descent
    fmincon_opts = optimoptions('fmincon', ...
        'Algorithm', 'sqp', ...
        'Display', 'none', ...
        'SpecifyObjectiveGradient', true, ...
        'MaxIterations', opts.gd_max_iter, ...
        'OptimalityTolerance', 1e-6, ...
        'StepTolerance', 1e-10);
    
    if opts.verbose
        fmincon_opts.Display = 'iter';
    end
    
    % Run gradient-based optimization
    [u_opt, fval, exitflag] = fmincon(fun, u0, [], [], [], [], lb, ub, [], fmincon_opts);
end

%% Generate candidate solution for SA
function u_candidate = generate_candidate(u_current, lb, ub, step_size)
    
    % Generate random perturbation
    dim = length(u_current);
    perturbation = step_size * randn(dim, 1);
    
    % Apply perturbation
    u_candidate = u_current + perturbation;
    
    % Enforce bounds with reflection
    u_candidate = max(lb, min(ub, u_candidate));
    
    % If still out of bounds, use random restart for that dimension
    out_of_bounds = u_candidate < lb | u_candidate > ub;
    if any(out_of_bounds)
        u_candidate(out_of_bounds) = lb(out_of_bounds) + ...
            (ub(out_of_bounds) - lb(out_of_bounds)) .* rand(sum(out_of_bounds), 1);
    end
end

%% Merge options structures
function merged = merge_options(default_opts, user_opts)
    merged = default_opts;
    if isempty(user_opts)
        return;
    end
    
    fields = fieldnames(user_opts);
    for i = 1:length(fields)
        merged.(fields{i}) = user_opts.(fields{i});
    end
end

%% Original STL cost function (unchanged)
function [cost, grad_u] = stl_cost_devec_pre(u, A, B, g, z0, phi_str, C, dim_u, N, AB_cells)
    u_mat = reshape(u, dim_u, N);
    dim_z = length(z0);
    Z = zeros(dim_z, N);
    
    % Step-by-step Koopman propagation
    z = z0;
    for k = 1:N
        z = A*z + B*u_mat(:,k);
        Z(:,k) = z;
    end
    
    % Project to original states
    X = C*Z;
    
    % Wrap into struct for STL
    dim_x = size(C,1);
    x_struct = struct();
    max_vars = min(dim_x,10);
    for j = 1:max_vars
        x_struct.(sprintf('x%d', j)) = X(j,:);
    end
    
    % Compute STL robustness and gradient
    [rob, grad_struct] = robustness_gradient_fd(phi_str, x_struct);
    
    % Convert grad_struct to matrix
    grad_x = zeros(dim_x, N);
    for j = 1:max_vars
        grad_x(j,:) = grad_struct.(sprintf('x%d', j));
    end
    
    % Gradient w.r.t lifted states
    grad_z = C' * grad_x;
    
    % Compute gradient w.r.t inputs using precomputed AB_cells
    grad_u_mat = zeros(dim_u, N);
    for j = 1:N
        for k = j:N
            grad_u_mat(:,j) = grad_u_mat(:,j) + (AB_cells{k-j+1})' * grad_z(:,k);
        end
    end
    
    grad_u = grad_u_mat(:);
    cost = rob;
end