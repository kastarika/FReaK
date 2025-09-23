%% Main script for Koopman model identification and comparison
% This script identifies Koopman models using multiple methods:
% 1. Optimization-based (fmincon with multi-step error)
% 2. Extended Dynamic Mode Decomposition (EDMD)
% 3. getautokoopman function
% 4. Comparison with ground truth

clear; clc; close all;

%% Algorithm settings
numFeat = 50;          % number of observables
l = 1;                 % lengthscale parameter
maxFunEval = 1000;     % max function evaluations for optimization
maxIter = 100;         % max iterations for optimization
output_dim = 5;        % number of outputs
input_dim = 2;         % number of inputs
multi_step = 5;        % number of steps for multi-step error

% Generate Random Fourier Feature observables
g = randomFourierFeatureObservables(numFeat, output_dim, l);

%% Generate or load trajectory data
L = 5;                % number of trajectories
T = 100;               % final time
time_step = 5;         % time step
steps = T / time_step + 1;
timestamps = (0:steps - 1)' * time_step;

% Initialize trajectory storage
trajectories = cell(L, 1);

% Generate trajectories (replace with your actual system)
% Using your cars model as example
kfmodel = modelCars2();
kfmodel.ak.nObs = numFeat;
kfmodel.ak.dt = time_step;
kfmodel.T = T;
x = stl('x',5);
eq = globally(x(5)-x(4)<=40,interval(0,100));
kfModel.spec = specification(eq,'logic');
kfModel.runs=1;
kfModel.verb=2;
kfModel.maxSims=300;
kfModel.spec_string=coraBreachConvert(eq);
[kfmodel,trainset,soln,specSolns,allData] = initialize(kfmodel);

for i = 1:L
    [tsim, x, u, ~, ~] = sampleSimulation(kfmodel);
    trajectories{i}.t = tsim;
    trajectories{i}.x = x;      % states
    trajectories{i}.u = u;      % inputs
end

%% Transform data using observable functions
fprintf('Transforming data using observable functions...\n');
lifted_data = liftTrajectories(trajectories, g);

%% Method 1: Optimization-based Koopman (fmincon with multi-step error)
fprintf('\nMethod 1: Optimization-based identification...\n');
tic;
sysOpt = identifyKoopmanOptimization(lifted_data, output_dim, multi_step, maxFunEval, maxIter);
time_opt = toc;
fprintf('Optimization time: %.2f seconds\n', time_opt);

%% Method 2: Extended DMD (EDMD)
fprintf('\nMethod 2: EDMD identification...\n');
tic;
sysEDMD = identifyKoopmanEDMD(lifted_data);
time_edmd = toc;
fprintf('EDMD time: %.2f seconds\n', time_edmd);

%% Method 3: getautokoopman
fprintf('\nMethod 3: getautokoopman...\n');
tic;
koopModel = getautokoopman(trajectories, kfmodel);
time_auto = toc;
fprintf('Autokoopman time: %.2f seconds\n', time_auto);

%% Simulate and compare all methods
fprintf('\nSimulating identified models...\n');
simResults = struct();

for i = 1:L
    % Initial condition in lifted space
    x0_lifted = g(trajectories{i}.x(1,:)');
    
    % Optimization-based simulation
    simOpts.x0 = x0_lifted;
    simOpts.tFinal = T;
    simOpts.u = trajectories{i}.u;
    [tOpt, xOpt] = simulateKoopman(sysOpt, simOpts);
    simResults.opt{i}.t = tOpt;
    simResults.opt{i}.x = xOpt;
    
    % EDMD simulation
    [tEDMD, xEDMD] = simulateKoopman(sysEDMD, simOpts);
    simResults.edmd{i}.t = tEDMD;
    simResults.edmd{i}.x = xEDMD;
end

%% Compute errors and statistics
errors = computeErrors(trajectories, simResults, output_dim);

%% Visualization
visualizeResults(trajectories, simResults, errors, output_dim);

%% Display summary
fprintf('\n========== SUMMARY ==========\n');
fprintf('Average RMSE (first %d outputs):\n', output_dim);
fprintf('Optimization-based: %.4f\n', errors.opt.avg_rmse);
fprintf('EDMD:              %.4f\n', errors.edmd.avg_rmse);
fprintf('\nComputation times:\n');
fprintf('Optimization: %.2f s\n', time_opt);
fprintf('EDMD:        %.2f s\n', time_edmd);
fprintf('Autokoopman: %.2f s\n', time_auto);

%% ==================== HELPER FUNCTIONS ====================

function g = randomFourierFeatureObservables(numFeat, dim, l)
    % Generate Random Fourier Feature observables cos(w'*x + u)
    w = normrnd(0, l^2, numFeat, dim);
    u = 2*pi*rand(numFeat, 1);
    g = @(x) [x; sqrt(2)*cos(w*x + u)];
end

function lifted_data = liftTrajectories(trajectories, g)
    % Lift trajectories to observable space
    L = length(trajectories);
    lifted_data = cell(L, 1);
    
    for i = 1:L
        t = trajectories{i}.t;
        x = trajectories{i}.x;
        u = trajectories{i}.u;
        
        % Lift states
        x_lifted = zeros(size(x,1), length(g(x(1,:)')));
        for j = 1:size(x, 1)
            x_lifted(j, :) = g(x(j,:)')';
        end
        
        lifted_data{i}.t = t;
        lifted_data{i}.x = x_lifted;
        lifted_data{i}.u = u;
    end
end

function sys = identifyKoopmanOptimization(data, p, multi_step, maxFunEval, maxIter)
    % Identify Koopman operator using optimization with multi-step error
    
    % Convert to uniform time step
    dt = computeAverageTimeStep(data);
    data = uniformTimeStep(data, dt);
    
    % Get dimensions
    n = size(data{1}.x, 2);  % lifted dimension
    
    % Check if inputs exist
    if isfield(data{1}, 'u') && ~isempty(data{1}.u)
        m = size(data{1}.u, 2);  % input dimension
    else
        m = 0;
    end
    
    % Initial guess using EDMD
    sysInit = identifyKoopmanEDMD(data);
    
    % Flatten initial matrices
    if m > 0
        x0 = [sysInit.A(:); sysInit.B(:); sysInit.c(:)];
    else
        x0 = [sysInit.A(:); sysInit.c(:)];
    end
    
    % Create figure for live plotting
    figure('Name', 'Optimization Progress', 'Position', [100, 100, 1200, 400]);
    
    % Optimization options with live display
    options = optimoptions('fmincon', ...
        'SpecifyObjectiveGradient', false, ...
        'Display', 'iter-detailed', ...  % Shows detailed output at each iteration
        'MaxIterations', maxIter, ...
        'MaxFunctionEvaluations', maxFunEval, ...
        'OptimalityTolerance', 1e-6, ...
        'StepTolerance', 1e-8, ...
        'PlotFcn', {@optimplotfval, @optimplotfirstorderopt, @optimplotstepsize}, ... % Live plots
        'OutputFcn', @(x, optimValues, state) outputFun(x, optimValues, state, data, p, n, m));  % Custom live output
    
    % Run optimization
    fprintf('\n========== STARTING OPTIMIZATION ==========\n');
    fprintf('Initial cost: %.6f\n', multiStepCost(x0, data, p, multi_step, n, m));
    fprintf('Parameters: %d dimensions, %d trajectories, %d multi-steps\n\n', length(x0), length(data), multi_step);
    
    x_opt = fmincon(@(x) multiStepCost(x, data, p, multi_step, n, m), ...
                    x0, [], [], [], [], [], [], [], options);
    
    fprintf('\n========== OPTIMIZATION COMPLETE ==========\n');
    fprintf('Final cost: %.6f\n', multiStepCost(x_opt, data, p, multi_step, n, m));
    
    % Reconstruct matrices
    idx = 0;
    A = reshape(x_opt(1:n*n), [n, n]);
    idx = n*n;
    
    if m > 0
        B = reshape(x_opt(idx+1:idx+n*m), [n, m]);
        idx = idx + n*m;
    else
        B = [];
    end
    
    c = x_opt(idx+1:end);
    
    % Create system struct
    sys.A = A;
    sys.B = B;
    sys.c = c(:);  % Ensure column vector
    sys.dt = dt;
    sys.n = n;
    sys.m = m;
    sys.p = p;
end

function stop = outputFun(x, optimValues, state, data, p, n, m)
    % Custom output function for live display during optimization
    stop = false;
    
    persistent iteration_data cost_history
    
    switch state
        case 'init'
            % Initialize
            iteration_data = [];
            cost_history = [];
            fprintf('Optimization initialized. Starting iterations...\n');
            fprintf('%-10s %-15s %-15s %-15s %-15s\n', ...
                    'Iter', 'Cost', 'Step Size', '1st Order Opt', 'Improvement');
            fprintf('%s\n', repmat('-', 80, 1));
            
        case 'iter'
            % Store history
            if isempty(cost_history)
                cost_history = optimValues.fval;
                improvement = 0;
            else
                improvement = cost_history(end) - optimValues.fval;
                cost_history = [cost_history, optimValues.fval];
            end
            
            % Display iteration info
            fprintf('%-10d %-15.6e %-15.6e %-15.6e %-15.6e\n', ...
                    optimValues.iteration, ...
                    optimValues.fval, ...
                    optimValues.stepsize, ...
                    optimValues.firstorderopt, ...
                    improvement);
            
            % Update live plot in subplot 4 (custom plot)
            if optimValues.iteration > 0
                subplot(1,3,3);
                plot(0:optimValues.iteration, [cost_history(1), cost_history], 'b-', 'LineWidth', 2);
                xlabel('Iteration');
                ylabel('Cost');
                title('Cost Reduction Progress');
                grid on;
                drawnow;  % Force MATLAB to update the display immediately
            end
            
        case 'done'
            fprintf('%s\n', repmat('=', 80, 1));
            fprintf('Optimization finished!\n');
            if ~isempty(cost_history)
                fprintf('Total improvement: %.6e (%.2f%% reduction)\n', ...
                        cost_history(1) - cost_history(end), ...
                        100*(cost_history(1) - cost_history(end))/cost_history(1));
            end
    end
end

function cost = multiStepCost(x, data, p, multi_step, n, m)
    % Compute multi-step prediction error (without gradient for stability)
    
    % Reshape parameters
    A = reshape(x(1:n*n), [n, n]);
    if m > 0
        B = reshape(x(n*n+1:n*n+n*m), [n, m]);
        c = x(n*n+n*m+1:end);
    else
        B = [];
        c = x(n*n+1:end);
    end
    c = c(:);  % Ensure column vector
    
    cost = 0;
    
    % Loop over trajectories
    for i = 1:length(data)
        N = size(data{i}.x, 1);
        
        % Loop over time windows
        for j = 1:N-multi_step
            % Initial state
            x_curr = data{i}.x(j, :)';
            
            % Multi-step prediction
            for k = 1:multi_step
                if m > 0 && isfield(data{i}, 'u') && ~isempty(data{i}.u)
                    if j+k-1 <= size(data{i}.u, 1)
                        u_curr = data{i}.u(j+k-1, :)';
                    else
                        u_curr = zeros(m, 1);
                    end
                    % Predict next state with input
                    x_pred = A * x_curr + B * u_curr + c;
                else
                    % Predict next state without input
                    x_pred = A * x_curr + c;
                end
                
                % True next state
                x_true = data{i}.x(j+k, :)';
                
                % Error (only first p outputs)
                err = x_pred(1:p) - x_true(1:p);
                cost = cost + 0.5 * sum(err.^2);
                
                % Update state for next step
                x_curr = x_pred;
            end
        end
    end
end

function sys = identifyKoopmanEDMD(data)
    % Extended Dynamic Mode Decomposition (EDMD) with inputs
    
    % Convert to uniform time step
    dt = computeAverageTimeStep(data);
    data = uniformTimeStep(data, dt);
    
    % Collect snapshot pairs
    X = [];  % Current states
    Y = [];  % Next states
    U = [];  % Inputs
    
    for i = 1:length(data)
        X = [X, data{i}.x(1:end-1, :)'];
        Y = [Y, data{i}.x(2:end, :)'];
        if isfield(data{i}, 'u') && ~isempty(data{i}.u)
            % Handle different input sizes
            if size(data{i}.u, 1) >= size(data{i}.x, 1) - 1
                U = [U, data{i}.u(1:size(data{i}.x, 1)-1, :)'];
            else
                % Pad with zeros if needed
                u_padded = [data{i}.u; zeros(size(data{i}.x, 1)-1-size(data{i}.u, 1), size(data{i}.u, 2))];
                U = [U, u_padded'];
            end
        end
    end
    
    % Augment with inputs and bias
    if ~isempty(U)
        Psi = [X; U; ones(1, size(X, 2))];  % [x; u; 1]
    else
        Psi = [X; ones(1, size(X, 2))];     % [x; 1]
    end
    
    % EDMD: Y = K * Psi, where K contains [A, B, c]
    % Use SVD-based pseudoinverse for better numerical stability
    K = Y / Psi;  % Equivalent to Y * pinv(Psi) but more stable
    
    % Extract matrices
    n = size(X, 1);
    A = K(:, 1:n);
    
    if ~isempty(U)
        m = size(U, 1);
        B = K(:, n+1:n+m);
        c = K(:, n+m+1);
    else
        B = [];
        c = K(:, n+1);
        m = 0;
    end
    
    % Create system struct
    sys.A = A;
    sys.B = B;
    sys.c = c(:);  % Ensure column vector
    sys.dt = dt;
    sys.n = n;
    sys.m = m;
end

function [t, x] = simulateKoopman(sys, simOpts)
    % Simulate Koopman system
    
    x0 = simOpts.x0;
    tFinal = simOpts.tFinal;
    u_traj = simOpts.u;
    
    % Time vector
    t = 0:sys.dt:tFinal;
    N = length(t);
    
    % Initialize state trajectory
    x = zeros(N, length(x0));
    x(1, :) = x0';
    
    % Simulate
    for k = 1:N-1
        if ~isempty(u_traj) && ~isempty(sys.B)
            if k <= size(u_traj, 1)
                u = u_traj(k, :)';
            else
                u = zeros(sys.m, 1);
            end
        else
            u = zeros(max(sys.m, 1), 1);
        end
        
        x_curr = x(k, :)';
        
        % Handle case where B might be empty
        if ~isempty(sys.B)
            x_next = sys.A * x_curr + sys.B * u + sys.c;
        else
            x_next = sys.A * x_curr + sys.c;
        end
        
        x(k+1, :) = x_next';
    end
end

function dt = computeAverageTimeStep(data)
    % Compute average time step
    dt_sum = 0;
    count = 0;
    
    for i = 1:length(data)
        dt_sum = dt_sum + sum(diff(data{i}.t));
        count = count + length(data{i}.t) - 1;
    end
    
    dt = dt_sum / count;
end

function data = uniformTimeStep(data, dt)
    % Convert data to uniform time step
    for i = 1:length(data)
        % Check if already uniform
        time_diffs = diff(data{i}.t);
        if all(abs(time_diffs - dt) < 1e-10)
            continue;  % Already uniform
        end
        
        % Create uniform time vector
        t_uniform = (data{i}.t(1):dt:data{i}.t(end))';
        
        % Remove any duplicate time points
        [t_unique, idx_unique] = unique(data{i}.t);
        
        % Interpolate states
        data{i}.x = interp1(t_unique, data{i}.x(idx_unique, :), t_uniform, 'linear', 'extrap');
        
        % Interpolate inputs if present
        if isfield(data{i}, 'u') && ~isempty(data{i}.u)
            % Handle different input dimensions
            if size(data{i}.u, 1) == length(data{i}.t)
                % Input for each time point
                data{i}.u = interp1(t_unique, data{i}.u(idx_unique, :), t_uniform, 'linear', 'extrap');
            elseif size(data{i}.u, 1) == length(data{i}.t) - 1
                % Input between time points (common in discrete systems)
                t_u = data{i}.t(1:end-1);
                [t_u_unique, idx_u_unique] = unique(t_u);
                % Interpolate to new time points (excluding last)
                if length(t_uniform) > 1
                    data{i}.u = interp1(t_u_unique, data{i}.u(idx_u_unique, :), t_uniform(1:end-1), 'linear', 'extrap');
                end
            else
                % If dimensions don't match, keep original or resize
                warning('Input dimensions do not match time vector for trajectory %d', i);
                % Try to resize to match new time vector
                if size(data{i}.u, 1) > 0
                    data{i}.u = interp1(1:size(data{i}.u, 1), data{i}.u, ...
                        linspace(1, size(data{i}.u, 1), length(t_uniform)-1), 'linear');
                end
            end
        end
        
        data{i}.t = t_uniform;
    end
end

function errors = computeErrors(trajectories, simResults, p)
    % Compute errors for each method
    
    L = length(trajectories);
    
    % Optimization errors
    errors.opt.rmse = zeros(L, 1);
    for i = 1:L
        true_x = trajectories{i}.x(:, 1:p);
        pred_x = simResults.opt{i}.x(:, 1:p);
        errors.opt.rmse(i) = sqrt(mean((true_x(:) - pred_x(:)).^2));
    end
    errors.opt.avg_rmse = mean(errors.opt.rmse);
    
    % EDMD errors
    errors.edmd.rmse = zeros(L, 1);
    for i = 1:L
        true_x = trajectories{i}.x(:, 1:p);
        pred_x = simResults.edmd{i}.x(:, 1:p);
        errors.edmd.rmse(i) = sqrt(mean((true_x(:) - pred_x(:)).^2));
    end
    errors.edmd.avg_rmse = mean(errors.edmd.rmse);
end

function visualizeResults(trajectories, simResults, errors, p)
    % Visualize comparison results
    
    % Plot sample trajectories
    num_plots = min(3, length(trajectories));
    
    for i = 1:num_plots
        figure('Position', [100, 100, 1200, 600]);
        
        for j = 1:min(p, 5)  % Plot up to 5 outputs
            subplot(2, 3, j);
            hold on;
            
            % Ground truth
            plot(trajectories{i}.t, trajectories{i}.x(:, j), 'k-', ...
                 'LineWidth', 2, 'DisplayName', 'Ground Truth');
            
            % Optimization
            plot(simResults.opt{i}.t, simResults.opt{i}.x(:, j), 'b--', ...
                 'LineWidth', 1.5, 'DisplayName', 'Optimization');
            
            % EDMD
            plot(simResults.edmd{i}.t, simResults.edmd{i}.x(:, j), 'r:', ...
                 'LineWidth', 1.5, 'DisplayName', 'EDMD');
            
            xlabel('Time');
            ylabel(sprintf('x_%d', j));
            title(sprintf('Output %d - Trajectory %d', j, i));
            legend('Location', 'best');
            grid on;
            hold off;
        end
        
        % Add error information
        subplot(2, 3, 6);
        axis off;
        text(0.1, 0.7, sprintf('Trajectory %d Errors:', i), 'FontWeight', 'bold');
        text(0.1, 0.5, sprintf('Opt RMSE: %.4f', errors.opt.rmse(i)));
        text(0.1, 0.3, sprintf('EDMD RMSE: %.4f', errors.edmd.rmse(i)));
    end
    
    % Error comparison bar plot
    figure('Position', [100, 100, 600, 400]);
    bar([errors.opt.avg_rmse, errors.edmd.avg_rmse]);
    set(gca, 'XTickLabel', {'Optimization', 'EDMD'});
    ylabel('Average RMSE');
    title('Model Comparison - Average RMSE');
    grid on;
end