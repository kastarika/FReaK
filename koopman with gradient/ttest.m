%% Complete Example: Gradient-based Koopman Identification
% This example compares DMD vs Gradient Descent for Koopman identification
% using multi-step prediction loss

clear; clc; close all;

%% 1. Setup: Van der Pol Oscillator
fprintf('=== Koopman Identification: DMD vs Gradient Descent ===\n\n');

% System dynamics
f_vdp = @(t,x) [x(2); (1-x(1)^2)*x(2)-x(1)];

% Parameters
dt = 0.01;
T_train = 10;      % Training time
T_test = 5;        % Testing time
n_traj_train = 10; % Number of training trajectories
n_features = 50;   % Number of random Fourier features

%% 2. Create Observable Functions
fprintf('Creating Random Fourier Features...\n');
fprintf('  Original dimension: 2\n');
fprintf('  Lifted dimension: %d\n\n', n_features + 2);

% Random Fourier Features
lengthscale = 1;
W = randn(n_features, 2) / lengthscale;
b = 2*pi*rand(n_features, 1);

% Observable function and its inverse
g = @(x) [x; sqrt(2)*cos(W*x + b)];
g_inv = @(z) z(1:2);  % Extract original coordinates

n_obs = n_features + 2;

%% 3. Generate Training Data
fprintf('Generating training trajectories...\n');

t_train = 0:dt:T_train;
X_traj_train = cell(n_traj_train, 1);
U_traj_train = [];  % No control inputs for Van der Pol

for i = 1:n_traj_train
    % Random initial condition
    x0 = [3*rand-1.5; 3*rand-1.5];
    
    % Simulate
    [~, x] = ode45(f_vdp, t_train, x0);
    
    % Transform to observable space
    X_obs = zeros(n_obs, length(t_train));
    for j = 1:length(t_train)
        X_obs(:, j) = g(x(j,:)');
    end
    
    X_traj_train{i} = X_obs;
    
    fprintf('  Trajectory %2d: x0 = [%5.2f, %5.2f], %d time steps\n', ...
        i, x0(1), x0(2), size(X_obs, 2));
end

%% 4. Method 1: Standard DMD
fprintf('\n--- Method 1: Dynamic Mode Decomposition ---\n');

% Prepare data for DMD
X_dmd = [];
Y_dmd = [];

for i = 1:n_traj_train
    X_dmd = [X_dmd, X_traj_train{i}(:, 1:end-1)];
    Y_dmd = [Y_dmd, X_traj_train{i}(:, 2:end)];
end

% Compute DMD
tic;
A_dmd = Y_dmd / X_dmd;  % Least squares solution
time_dmd = toc;

fprintf('  Training time: %.2f seconds\n', time_dmd);
fprintf('  Operator size: %d x %d\n', size(A_dmd, 1), size(A_dmd, 2));

% Check eigenvalues
eigs_dmd = eig(A_dmd);
fprintf('  Max eigenvalue magnitude: %.4f\n', max(abs(eigs_dmd)));

%% 5. Method 2: Gradient Descent with Multi-Step Loss
fprintf('\n--- Method 2: Gradient Descent (Multi-Step) ---\n');

% Different multi-step horizons to test
horizons = [5, 10, 20];
A_grad = cell(length(horizons), 1);
history = cell(length(horizons), 1);

for h = 1:length(horizons)
    n_steps = horizons(h);
    fprintf('\n  Horizon: %d steps\n', n_steps);
    
    tic;
    [A_grad{h}, ~, history{h}] = identifyKoopmanGradient(X_traj_train, [], ...
        'n_steps', n_steps, ...
        'learning_rate', 0.001, ...
        'max_iter', 500, ...
        'optimizer', 'adam', ...
        'init_method', 'dmd', ...
        'regularization', 1e-5, ...
        'verbose', false, ...
        'plot_loss', false);
    time_grad = toc;
    
    fprintf('    Training time: %.2f seconds\n', time_grad);
    fprintf('    Final loss: %.6e\n', history{h}.loss(end));
    
    % Check eigenvalues
    eigs_grad = eig(A_grad{h});
    fprintf('    Max eigenvalue magnitude: %.4f\n', max(abs(eigs_grad)));
end

%% 6. Generate Test Trajectories
fprintf('\n=== Testing Performance ===\n');

% Test initial conditions
test_x0 = [
    2.0,  0.0;
    1.5,  1.0;
   -1.0,  1.5;
    0.5, -2.0;
];

n_test = size(test_x0, 1);
t_test = 0:dt:T_test;

%% 7. Compare Predictions
fprintf('\nComputing prediction errors...\n');

% Store errors
errors_dmd = zeros(n_test, length(t_test));
errors_grad = cell(length(horizons), 1);
for h = 1:length(horizons)
    errors_grad{h} = zeros(n_test, length(t_test));
end

% For each test trajectory
for test_idx = 1:n_test
    % True trajectory
    [~, x_true] = ode45(f_vdp, t_test, test_x0(test_idx, :)');
    
    % Initial condition in observable space
    z0 = g(test_x0(test_idx, :)');
    
    % DMD prediction
    z_dmd = zeros(n_obs, length(t_test));
    z_dmd(:, 1) = z0;
    for k = 2:length(t_test)
        z_dmd(:, k) = A_dmd * z_dmd(:, k-1);
    end
    x_dmd = g_inv(z_dmd)';
    
    % Gradient predictions
    x_grad = cell(length(horizons), 1);
    for h = 1:length(horizons)
        z_grad = zeros(n_obs, length(t_test));
        z_grad(:, 1) = z0;
        for k = 2:length(t_test)
            z_grad(:, k) = A_grad{h} * z_grad(:, k-1);
        end
        x_grad{h} = g_inv(z_grad)';
    end
    
    % Compute errors
    errors_dmd(test_idx, :) = vecnorm(x_dmd - x_true, 2, 2)';
    for h = 1:length(horizons)
        errors_grad{h}(test_idx, :) = vecnorm(x_grad{h} - x_true, 2, 2)';
    end
end

% Average errors
mean_error_dmd = mean(errors_dmd, 1);
mean_errors_grad = zeros(length(horizons), length(t_test));
for h = 1:length(horizons)
    mean_errors_grad(h, :) = mean(errors_grad{h}, 1);
end

%% 8. Visualize Results

% Figure 1: Training Loss Evolution
figure('Position', [100, 100, 1200, 400]);
colors = {'b', 'r', 'g'};
for h = 1:length(horizons)
    subplot(1, 3, h);
    plot(history{h}.loss, colors{h}, 'LineWidth', 1.5);
    xlabel('Iteration');
    ylabel('Loss');
    title(sprintf('Training Loss (Horizon = %d)', horizons(h)));
    grid on;
    set(gca, 'YScale', 'log');
end
sgtitle('Gradient Descent Training Progress');

% Figure 2: Prediction Error Comparison
figure('Position', [100, 500, 800, 600]);
plot(t_test, mean_error_dmd, 'k-', 'LineWidth', 2, 'DisplayName', 'DMD');
hold on;
for h = 1:length(horizons)
    plot(t_test, mean_errors_grad(h, :), '--', ...
        'Color', colors{h}, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Gradient (h=%d)', horizons(h)));
end
xlabel('Time');
ylabel('Mean Prediction Error');
title('Prediction Error: DMD vs Gradient Descent');
legend('Location', 'northwest');
grid on;

% Figure 3: Phase Portrait Comparison (best gradient model)
[~, best_h] = min(mean(mean_errors_grad, 2));
figure('Position', [950, 500, 1200, 500]);

test_case = 1;  % Which test trajectory to plot
[~, x_true] = ode45(f_vdp, t_test, test_x0(test_case, :)');

% DMD prediction
z0 = g(test_x0(test_case, :)');
z_dmd = zeros(n_obs, length(t_test));
z_dmd(:, 1) = z0;
for k = 2:length(t_test)
    z_dmd(:, k) = A_dmd * z_dmd(:, k-1);
end
x_dmd = g_inv(z_dmd)';

% Best gradient prediction
z_grad = zeros(n_obs, length(t_test));
z_grad(:, 1) = z0;
for k = 2:length(t_test)
    z_grad(:, k) = A_grad{best_h} * z_grad(:, k-1);
end
x_grad = g_inv(z_grad)';

% Plot phase portraits
subplot(1, 2, 1);
plot(x_true(:,1), x_true(:,2), 'k-', 'LineWidth', 2, 'DisplayName', 'True');
hold on;
plot(x_dmd(:,1), x_dmd(:,2), 'b--', 'LineWidth', 1.5, 'DisplayName', 'DMD');
plot(test_x0(test_case, 1), test_x0(test_case, 2), 'go', ...
    'MarkerSize', 8, 'MarkerFaceColor', 'g', 'DisplayName', 'Start');
xlabel('x_1'); ylabel('x_2');
title('DMD Prediction');
legend('Location', 'best');
grid on;
axis equal;
xlim([-3, 3]); ylim([-3, 3]);

subplot(1, 2, 2);
plot(x_true(:,1), x_true(:,2), 'k-', 'LineWidth', 2, 'DisplayName', 'True');
hold on;
plot(x_grad(:,1), x_grad(:,2), 'r--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('Gradient (h=%d)', horizons(best_h)));
plot(test_x0(test_case, 1), test_x0(test_case, 2), 'go', ...
    'MarkerSize', 8, 'MarkerFaceColor', 'g', 'DisplayName', 'Start');
xlabel('x_1'); ylabel('x_2');
title(sprintf('Gradient Descent (Best: h=%d)', horizons(best_h)));
legend('Location', 'best');
grid on;
axis equal;
xlim([-3, 3]); ylim([-3, 3]);

sgtitle('Phase Portrait Comparison: DMD vs Best Gradient Model');

%% 9. Performance Summary
fprintf('\n=== Performance Summary ===\n');
fprintf('%-20s | %-15s | %-15s | %-15s\n', 'Method', 'Mean Error', 'Max Error', 'Final Error');
fprintf('%s\n', repmat('-', 70, 1));

% DMD performance
fprintf('%-20s | %15.6f | %15.6f | %15.6f\n', ...
    'DMD', mean(mean_error_dmd), max(mean_error_dmd), mean_error_dmd(end));

% Gradient descent performance
for h = 1:length(horizons)
    fprintf('%-20s | %15.6f | %15.6f | %15.6f\n', ...
        sprintf('Gradient (h=%d)', horizons(h)), ...
        mean(mean_errors_grad(h,:)), ...
        max(mean_errors_grad(h,:)), ...
        mean_errors_grad(h,end));
end

fprintf('\n=== Complete! ===\n');

%% MAIN FUNCTION: identifyKoopmanGradient
function [A, B, history] = identifyKoopmanGradient(X_traj, U_traj, varargin)
% IDENTIFYKOOPMANGRADIENT Identifies Koopman operator using gradient descent
% with multi-step prediction loss

% Parse inputs
p = inputParser;
addRequired(p, 'X_traj', @(x) iscell(x));
addOptional(p, 'U_traj', [], @(x) iscell(x) || isempty(x));
addParameter(p, 'n_steps', 10, @(x) x > 0);
addParameter(p, 'learning_rate', 0.01, @(x) x > 0);
addParameter(p, 'max_iter', 1000, @(x) x > 0);
addParameter(p, 'tol', 1e-6, @(x) x > 0);
addParameter(p, 'batch_size', inf, @(x) x > 0);
addParameter(p, 'momentum', 0.9, @(x) x >= 0 && x < 1);
addParameter(p, 'regularization', 1e-4, @(x) x >= 0);
addParameter(p, 'init_method', 'dmd', @ischar);
addParameter(p, 'optimizer', 'adam', @ischar);
addParameter(p, 'lr_decay', 0.99, @(x) x > 0 && x <= 1);
addParameter(p, 'verbose', true, @islogical);
addParameter(p, 'plot_loss', true, @islogical);

parse(p, X_traj, varargin{:});
opts = p.Results;
U_traj = p.Results.U_traj;

% Get dimensions
n_traj = length(X_traj);
n = size(X_traj{1}, 1);  % Number of observables
has_inputs = ~isempty(U_traj);

if has_inputs
    m = size(U_traj{1}, 1);  % Number of inputs
else
    m = 0;
end

% Validate trajectories
for i = 1:n_traj
    if size(X_traj{i}, 1) ~= n
        error('All trajectories must have the same number of observables');
    end
    if has_inputs && size(U_traj{i}, 2) ~= size(X_traj{i}, 2) - 1
        error('Input trajectory %d has wrong dimensions', i);
    end
end

% Initialize A and B
[A, B] = initialize_matrices(X_traj, U_traj, n, m, opts.init_method);

% Prepare for optimization
history.loss = [];
history.grad_norm = [];
history.learning_rate = [];

% Initialize optimizer state
opt_state = init_optimizer(opts.optimizer, n, m);

% Main optimization loop
lr = opts.learning_rate;
best_loss = inf;
best_A = A;
best_B = B;
patience = 50;
no_improve_count = 0;

for iter = 1:opts.max_iter
    % Get batch of trajectories
    if opts.batch_size < n_traj
        batch_idx = randperm(n_traj, min(opts.batch_size, n_traj));
    else
        batch_idx = 1:n_traj;
    end
    
    % Compute loss and gradients
    [loss, grad_A, grad_B] = compute_multistep_loss_and_gradient(...
        A, B, X_traj(batch_idx), U_traj(batch_idx), opts.n_steps, opts.regularization);
    
    % Update parameters using chosen optimizer
    [A, B, opt_state] = update_parameters(...
        A, B, grad_A, grad_B, lr, opts.optimizer, opt_state, opts.momentum);
    
    % Store history
    history.loss(iter) = loss;
    history.grad_norm(iter) = norm([grad_A(:); grad_B(:)]);
    history.learning_rate(iter) = lr;
    
    % Update best model
    if loss < best_loss
        best_loss = loss;
        best_A = A;
        best_B = B;
        no_improve_count = 0;
    else
        no_improve_count = no_improve_count + 1;
    end
    
    % Learning rate decay
    if mod(iter, 50) == 0
        lr = lr * opts.lr_decay;
    end
    
    % Check convergence
    if iter > 10 && abs(history.loss(iter) - history.loss(iter-1)) < opts.tol
        break;
    end
    
    % Early stopping
    if no_improve_count > patience
        break;
    end
end

% Return best model
A = best_A;
B = best_B;

end

%% Helper Function: Initialize Matrices
function [A, B] = initialize_matrices(X_traj, U_traj, n, m, method)

switch lower(method)
    case 'dmd'
        % Use DMD for initialization
        X = [];
        Y = [];
        U = [];
        
        for i = 1:length(X_traj)
            X = [X, X_traj{i}(:, 1:end-1)];
            Y = [Y, X_traj{i}(:, 2:end)];
            if ~isempty(U_traj)
                U = [U, U_traj{i}];
            end
        end
        
        if isempty(U)
            AB = Y / X;
            A = AB;
            B = zeros(n, 0);
        else
            XU = [X; U];
            AB = Y / XU;
            A = AB(:, 1:n);
            B = AB(:, n+1:end);
        end
        
    case 'random'
        % Random initialization
        A = randn(n, n) * 0.1;
        B = randn(n, m) * 0.1;
        
        % Make A slightly stable
        [V, D] = eig(A);
        D = D * 0.95;  % Scale eigenvalues
        A = real(V * D / V);
        
    case 'identity'
        % Identity initialization
        A = eye(n) + randn(n, n) * 0.01;
        B = randn(n, m) * 0.01;
        
    otherwise
        error('Unknown initialization method: %s', method);
end

end

%% Helper Function: Compute Loss and Gradient
function [loss, grad_A, grad_B] = compute_multistep_loss_and_gradient(A, B, X_traj, U_traj, n_steps, lambda)

n = size(A, 1);
m = size(B, 2);
n_traj = length(X_traj);
has_inputs = ~isempty(U_traj) && m > 0;

loss = 0;
grad_A = zeros(n, n);
grad_B = zeros(n, m);
n_samples = 0;

% For each trajectory
for traj_idx = 1:n_traj
    X = X_traj{traj_idx};
    T = size(X, 2);
    
    if has_inputs
        U = U_traj{traj_idx};
    end
    
    % For each starting point in the trajectory
    for t = 1:(T - n_steps)
        % Initial state
        x = X(:, t);
        
        % Forward propagation and store intermediate states
        x_pred = zeros(n, n_steps + 1);
        x_pred(:, 1) = x;
        
        for k = 1:n_steps
            if has_inputs && t+k-1 <= size(U, 2)
                u = U(:, t+k-1);
                x_pred(:, k+1) = A * x_pred(:, k) + B * u;
            else
                x_pred(:, k+1) = A * x_pred(:, k);
            end
        end
        
        % Compute loss (mean squared error over all steps)
        for k = 1:n_steps
            if t+k <= T
                error = x_pred(:, k+1) - X(:, t+k);
                loss = loss + 0.5 * norm(error)^2 / n_steps;
                n_samples = n_samples + 1;
                
                % Backpropagation through time
                delta = error / n_steps;
                
                % Gradient accumulation
                for j = k:-1:1
                    % Gradient w.r.t. A
                    grad_A = grad_A + delta * x_pred(:, j)';
                    
                    % Gradient w.r.t. B
                    if has_inputs && t+j-1 <= size(U, 2)
                        grad_B = grad_B + delta * U(:, t+j-1)';
                    end
                    
                    % Propagate error backward
                    delta = A' * delta;
                end
            end
        end
    end
end

% Normalize by number of samples
if n_samples > 0
    loss = loss / n_traj;
    grad_A = grad_A / n_samples;
    grad_B = grad_B / n_samples;
end

% Add L2 regularization
if lambda > 0
    loss = loss + 0.5 * lambda * (norm(A, 'fro')^2 + norm(B, 'fro')^2);
    grad_A = grad_A + lambda * A;
    grad_B = grad_B + lambda * B;
end

end

%% Helper Function: Initialize Optimizer
function opt_state = init_optimizer(optimizer, n, m)

switch lower(optimizer)
    case 'sgd'
        opt_state.v_A = zeros(n, n);  % Momentum
        opt_state.v_B = zeros(n, m);
        
    case 'adam'
        opt_state.m_A = zeros(n, n);  % First moment
        opt_state.v_A = zeros(n, n);  % Second moment
        opt_state.m_B = zeros(n, m);
        opt_state.v_B = zeros(n, m);
        opt_state.beta1 = 0.9;
        opt_state.beta2 = 0.999;
        opt_state.epsilon = 1e-8;
        opt_state.t = 0;
        
    case 'rmsprop'
        opt_state.v_A = zeros(n, n);  % Moving average of squared gradients
        opt_state.v_B = zeros(n, m);
        opt_state.beta = 0.9;
        opt_state.epsilon = 1e-8;
        
    otherwise
        error('Unknown optimizer: %s', optimizer);
end

end

%% Helper Function: Update Parameters
function [A, B, opt_state] = update_parameters(A, B, grad_A, grad_B, lr, optimizer, opt_state, momentum)

switch lower(optimizer)
    case 'sgd'
        % SGD with momentum
        opt_state.v_A = momentum * opt_state.v_A - lr * grad_A;
        opt_state.v_B = momentum * opt_state.v_B - lr * grad_B;
        A = A + opt_state.v_A;
        B = B + opt_state.v_B;
        
    case 'adam'
        % Adam optimizer
        opt_state.t = opt_state.t + 1;
        
        % Update biased first moment estimate
        opt_state.m_A = opt_state.beta1 * opt_state.m_A + (1 - opt_state.beta1) * grad_A;
        opt_state.m_B = opt_state.beta1 * opt_state.m_B + (1 - opt_state.beta1) * grad_B;
        
        % Update biased second raw moment estimate
        opt_state.v_A = opt_state.beta2 * opt_state.v_A + (1 - opt_state.beta2) * grad_A.^2;
        opt_state.v_B = opt_state.beta2 * opt_state.v_B + (1 - opt_state.beta2) * grad_B.^2;
        
        % Compute bias-corrected first moment estimate
        m_hat_A = opt_state.m_A / (1 - opt_state.beta1^opt_state.t);
        m_hat_B = opt_state.m_B / (1 - opt_state.beta1^opt_state.t);
        
        % Compute bias-corrected second raw moment estimate
        v_hat_A = opt_state.v_A / (1 - opt_state.beta2^opt_state.t);
        v_hat_B = opt_state.v_B / (1 - opt_state.beta2^opt_state.t);
        
        % Update parameters
        A = A - lr * m_hat_A ./ (sqrt(v_hat_A) + opt_state.epsilon);
        B = B - lr * m_hat_B ./ (sqrt(v_hat_B) + opt_state.epsilon);
        
    case 'rmsprop'
        % RMSprop optimizer
        opt_state.v_A = opt_state.beta * opt_state.v_A + (1 - opt_state.beta) * grad_A.^2;
        opt_state.v_B = opt_state.beta * opt_state.v_B + (1 - opt_state.beta) * grad_B.^2;
        
        A = A - lr * grad_A ./ (sqrt(opt_state.v_A) + opt_state.epsilon);
        B = B - lr * grad_B ./ (sqrt(opt_state.v_B) + opt_state.epsilon);
        
    otherwise
        error('Unknown optimizer: %s', optimizer);
end

end