%% Van der Pol Koopman Model Comparison: DMD vs Gradient Descent
% This code compares Koopman models learned via DMD and gradient descent
% for the Van der Pol oscillator system

clear; close all; clc;

%% Parameters
mu = 2.0;           % Van der Pol parameter
dt = 0.01;          % Time step
T_train = 10;       % Training time
T_test = 5;         % Test time
n_traj = 20;        % Number of training trajectories

% Initial conditions for training (random sampling)
rng(42); % For reproducibility
x0_train = 4 * (rand(2, n_traj) - 0.5); % Random ICs in [-2, 2]^2

% Define system dimensions
dim_x = 2;          % State dimension (Van der Pol is 2D)
dim_u = 1;          % Control dimension (autonomous system)

%% Generate Training Data
fprintf('Generating training data...\n');

% Collect training trajectories
X_data = [];
X_next_data = [];
U_data = [];

for i = 1:n_traj
    % Simulate Van der Pol oscillator
    [t, x_traj] = simulate_vanderpol(x0_train(:,i), [0, T_train], dt, mu);
    
    % Zero control input for autonomous system
    u_traj = zeros(1, length(t)-1);
    
    % Stack data (exclude last point for X_next)
    X_data = [X_data, x_traj(:, 1:end-1)];
    X_next_data = [X_next_data, x_traj(:, 2:end)];
    U_data = [U_data, u_traj];
end

fprintf('Training data: %d state snapshots\n', size(X_data, 2));

%% Define Random Fourier Features (RFF) Observable Functions
% RFF parameters
D_rff = 50;              % Number of random features (must be even)
sigma_rff = 1.0;         % RBF kernel bandwidth
rng(123);                % For reproducible random features

% Ensure D_rff is even
if mod(D_rff, 2) ~= 0
    D_rff = D_rff + 1;
    fprintf('Warning: D_rff adjusted to %d (must be even)\n', D_rff);
end

D_half = D_rff / 2;      % Half the number of features (integer)

% Generate random frequencies for RFF
W_rff = randn(dim_x, D_half) / sigma_rff;  % Random frequencies
b_rff = 2*pi*rand(1, D_half);              % Random phases

% RFF observable function
observe = @(x) rff_features(x, W_rff, b_rff);

function z = rff_features(x, W, b)
    % Random Fourier Features implementation
    % x: [dim_x, N] state matrix
    % W: [dim_x, D/2] random frequency matrix  
    % b: [1, D/2] random phase vector
    % Returns: [dim_x + D, N] lifted features
    
    if size(x, 1) == 1
        x = x(:);  % Ensure column vector for single point
    end
    
    % Compute random projections
    projections = W' * x + b';  % [D/2, N]
    
    % Compute cosine and sine features
    cos_features = cos(projections);  % [D/2, N]
    sin_features = sin(projections);  % [D/2, N]
    
    % Include original states and RFF features
    z = [x; cos_features; sin_features];  % [dim_x + D, N]
end

% Lift training data
Z_data = observe(X_data);
Z_next_data = observe(X_next_data);

dim_z = size(Z_data, 1);
dim_x = size(X_data, 1);
dim_u = size(U_data, 1);

fprintf('RFF parameters: D=%d, σ=%.2f\n', D_rff, sigma_rff);
fprintf('Lifted space dimension: %d (original: %d, RFF: %d)\n', dim_z, dim_x, D_rff);

%% Method 1: Dynamic Mode Decomposition (DMD)
fprintf('\n=== DMD Method ===\n');

% For autonomous system (no control), we have Z_next = A_dmd * Z
% Solve: Z_next = A_dmd * Z  =>  A_dmd = Z_next * pinv(Z)

tic;
A_dmd = Z_next_data * pinv(Z_data);
time_dmd = toc;

fprintf('DMD computation time: %.4f seconds\n', time_dmd);
fprintf('DMD A matrix condition number: %.2e\n', cond(A_dmd));

%% Method 2: Multi-Step Gradient Descent
fprintf('\n=== Multi-Step Gradient Descent Method ===\n');

% Initialize A matrix randomly
A_gd = 0.1 * randn(dim_z, dim_z);

% Multi-step gradient descent parameters
learning_rate = 1e-5;        % Reduced for RFF (higher dimensional)
max_iterations = 2000;       % More iterations for RFF
tolerance = 1e-8;
prediction_steps = 5;        % Number of steps to predict ahead
batch_size = min(100, floor(size(Z_data, 2) / prediction_steps));  % Smaller batches for RFF
step_weights = [1.0, 0.8, 0.6, 0.4, 0.2]; % Decreasing weights for future steps

% Prepare multi-step training data
fprintf('Preparing multi-step training sequences...\n');
Z_sequences = [];
target_sequences = [];

% Create sequences of length prediction_steps + 1
sequence_length = prediction_steps + 1;
n_sequences = size(Z_data, 2) - prediction_steps;

for i = 1:n_sequences
    if i + prediction_steps <= size(Z_data, 2)
        Z_sequences = cat(3, Z_sequences, Z_data(:, i:i+prediction_steps));
    end
end

fprintf('Created %d training sequences of length %d\n', size(Z_sequences, 3), sequence_length);

% Training loop
tic;
losses = zeros(max_iterations, 1);
step_losses = zeros(max_iterations, prediction_steps);

fprintf('Starting multi-step gradient descent optimization...\n');

for iter = 1:max_iterations
    % Random batch sampling of sequences
    n_available = size(Z_sequences, 3);
    batch_idx = randperm(n_available, min(batch_size, n_available));
    
    total_loss = 0;
    grad_A = zeros(size(A_gd));
    
    % Process each sequence in the batch
    for seq_idx = batch_idx
        Z_seq = Z_sequences(:, :, seq_idx);
        Z_init = Z_seq(:, 1);  % Initial state
        
        % Multi-step forward prediction
        Z_pred = zeros(dim_z, prediction_steps + 1);
        Z_pred(:, 1) = Z_init;
        
        % Forward pass: predict multiple steps
        for step = 1:prediction_steps
            Z_pred(:, step + 1) = A_gd * Z_pred(:, step);
        end
        
        % Compute multi-step loss with weighted contributions
        step_loss_current = zeros(1, prediction_steps);
        for step = 1:prediction_steps
            error_step = Z_pred(:, step + 1) - Z_seq(:, step + 1);
            weight = step_weights(min(step, length(step_weights)));
            step_loss = weight * 0.5 * mean(error_step.^2);
            step_loss_current(step) = step_loss;
            total_loss = total_loss + step_loss;
            
            % Compute gradient contribution for this step
            % Chain rule: dL/dA = sum over steps of weight * error * (A^(step-1) * Z_init)^T
            if step == 1
                Z_input = Z_init;
            else
                % Compute A^(step-1) * Z_init
                Z_input = Z_init;
                for p = 1:step-1
                    Z_input = A_gd * Z_input;
                end
            end
            
            grad_A = grad_A + weight * (error_step * Z_input') / length(batch_idx);
        end
        
        step_losses(iter, :) = step_losses(iter, :) + step_loss_current / length(batch_idx);
    end
    
    losses(iter) = total_loss / length(batch_idx);
    
    % Update A matrix
    A_gd = A_gd - learning_rate * grad_A;
    
    % Check convergence
    if iter > 1 && abs(losses(iter-1) - losses(iter)) < tolerance
        fprintf('Converged at iteration %d\n', iter);
        break;
    end
    
    % Adaptive learning rate
    if iter > 100 && losses(iter) > losses(iter-50)
        learning_rate = learning_rate * 0.995;
    end
    
    % Progress update
    if mod(iter, 250) == 0
        fprintf('Iter %d: Total Loss = %.6f, Step Losses = [%.4f, %.4f, %.4f, %.4f, %.4f], LR = %.2e\n', ...
            iter, losses(iter), step_losses(iter, 1), step_losses(iter, 2), ...
            step_losses(iter, 3), step_losses(iter, 4), step_losses(iter, 5), learning_rate);
        
        % Additional diagnostics for RFF
        if mod(iter, 500) == 0
            fprintf('  A matrix stats: mean=%.4f, std=%.4f, max=%.4f\n', ...
                mean(A_gd(:)), std(A_gd(:)), max(abs(A_gd(:))));
        end
    end
end

time_gd = toc;
losses = losses(1:iter);
step_losses = step_losses(1:iter, :);

fprintf('Multi-step gradient descent time: %.4f seconds\n', time_gd);
fprintf('Final total loss: %.6e\n', losses(end));
fprintf('Final step losses: [%.4f, %.4f, %.4f, %.4f, %.4f]\n', ...
    step_losses(end, 1), step_losses(end, 2), step_losses(end, 3), step_losses(end, 4), step_losses(end, 5));
fprintf('Multi-step GD A matrix condition number: %.2e\n', cond(A_gd));

%% Projection Matrix (from lifted space back to original space)
C = [eye(dim_x), zeros(dim_x, dim_z - dim_x)];

%% Test on New Trajectory
fprintf('\n=== Model Comparison ===\n');

% Test initial condition
x0_test = [1.5; 0.5];
[t_test, x_true] = simulate_vanderpol(x0_test, [0, T_test], dt, mu);

% DMD Prediction
z0_test = observe(x0_test);
z_dmd = zeros(dim_z, length(t_test));
z_dmd(:, 1) = z0_test;

for k = 2:length(t_test)
    z_dmd(:, k) = A_dmd * z_dmd(:, k-1);
end
x_dmd = C * z_dmd;

% Gradient Descent Prediction
z_gd = zeros(dim_z, length(t_test));
z_gd(:, 1) = z0_test;

for k = 2:length(t_test)
    z_gd(:, k) = A_gd * z_gd(:, k-1);
end
x_gd = C * z_gd;

%% Error Analysis
error_dmd = sqrt(sum((x_dmd - x_true).^2, 1));
error_gd = sqrt(sum((x_gd - x_true).^2, 1));

rmse_dmd = sqrt(mean(error_dmd.^2));
rmse_gd = sqrt(mean(error_gd.^2));

fprintf('RMSE DMD: %.6f\n', rmse_dmd);
fprintf('RMSE Gradient Descent (Multi-step): %.6f\n', rmse_gd);

%% Eigenvalue Analysis
fprintf('\n=== Eigenvalue Analysis ===\n');

% Compute eigenvalues
eigs_dmd = eig(A_dmd);
eigs_gd = eig(A_gd);

% Continuous-time eigenvalues (convert from discrete)
eigs_dmd_cont = log(eigs_dmd) / dt;
eigs_gd_cont = log(eigs_gd) / dt;

% Dominant eigenvalues (largest magnitude)
[~, idx_dmd] = sort(abs(eigs_dmd_cont), 'descend');
[~, idx_gd] = sort(abs(eigs_gd_cont), 'descend');

fprintf('Top 5 DMD eigenvalues (continuous): \n');
for i = 1:min(5, length(eigs_dmd_cont))
    fprintf('  λ_%d = %.4f + %.4fi\n', i, real(eigs_dmd_cont(idx_dmd(i))), imag(eigs_dmd_cont(idx_dmd(i))));
end

fprintf('Top 5 GD eigenvalues (continuous): \n');
for i = 1:min(5, length(eigs_gd_cont))
    fprintf('  λ_%d = %.4f + %.4fi\n', i, real(eigs_gd_cont(idx_gd(i))), imag(eigs_gd_cont(idx_gd(i))));
end

%% Visualization
create_comparison_plots(t_test, x_true, x_dmd, x_gd, error_dmd, error_gd, losses, ...
                       eigs_dmd_cont, eigs_gd_cont, mu);

%% Summary
fprintf('\n=== SUMMARY ===\n');
fprintf('Training trajectories: %d\n', n_traj);
fprintf('Training time: %.1f s\n', T_train);
fprintf('Test time: %.1f s\n', T_test);
fprintf('Lifted dimension: %d (RFF: %d features, σ=%.2f)\n', dim_z, D_rff, sigma_rff);
fprintf('DMD RMSE: %.6f (computed in %.4f s)\n', rmse_dmd, time_dmd);
fprintf('Multi-step GD (RFF) RMSE:  %.6f (computed in %.4f s)\n', rmse_gd, time_gd);
fprintf('Winner: %s\n', ternary(rmse_dmd < rmse_gd, 'DMD', 'Multi-step GD with RFF'));

%% Helper Functions

function [t, x] = simulate_vanderpol(x0, tspan, dt, mu)
    % Simulate Van der Pol oscillator using ODE45
    options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
    
    vanderpol_ode = @(t, x) [x(2); mu*(1-x(1)^2)*x(2) - x(1)];
    
    [t_ode, x_ode] = ode45(vanderpol_ode, tspan, x0, options);
    
    % Interpolate to uniform grid
    t = tspan(1):dt:tspan(2);
    x = interp1(t_ode, x_ode, t, 'pchip')';
end

function result = ternary(condition, true_val, false_val)
    % Ternary operator for MATLAB
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

function create_comparison_plots(t, x_true, x_dmd, x_gd, error_dmd, error_gd, losses, eigs_dmd, eigs_gd, mu)
    % Create comprehensive comparison plots
    
    figure('Position', [100, 100, 1400, 900]);
    
    % Phase portraits
    subplot(2,4,1);
    plot(x_true(1,:), x_true(2,:), 'k-', 'LineWidth', 2); hold on;
    plot(x_dmd(1,:), x_dmd(2,:), 'r--', 'LineWidth', 1.5);
    plot(x_gd(1,:), x_gd(2,:), 'b:', 'LineWidth', 1.5);
    plot(x_true(1,1), x_true(2,1), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
    xlabel('x_1'); ylabel('x_2');
    title(sprintf('Phase Portrait (μ=%.1f)', mu));
    legend('True', 'DMD', 'Multi-step GD', 'IC', 'Location', 'best');
    grid on; axis equal;
    
    % Time series - x1
    subplot(2,4,2);
    plot(t, x_true(1,:), 'k-', 'LineWidth', 2); hold on;
    plot(t, x_dmd(1,:), 'r--', 'LineWidth', 1.5);
    plot(t, x_gd(1,:), 'b:', 'LineWidth', 1.5);
    xlabel('Time'); ylabel('x_1');
    title('State x_1 vs Time');
    legend('True', 'DMD', 'Multi-step GD', 'Location', 'best');
    grid on;
    
    % Time series - x2  
    subplot(2,4,3);
    plot(t, x_true(2,:), 'k-', 'LineWidth', 2); hold on;
    plot(t, x_dmd(2,:), 'r--', 'LineWidth', 1.5);
    plot(t, x_gd(2,:), 'b:', 'LineWidth', 1.5);
    xlabel('Time'); ylabel('x_2');
    title('State x_2 vs Time');
    legend('True', 'DMD', 'Multi-step GD', 'Location', 'best');
    grid on;
    
    % Error comparison
    subplot(2,4,4);
    semilogy(t, error_dmd, 'r-', 'LineWidth', 1.5); hold on;
    semilogy(t, error_gd, 'b-', 'LineWidth', 1.5);
    xlabel('Time'); ylabel('||Error||_2');
    title('Prediction Error');
    legend('DMD', 'Multi-step GD', 'Location', 'best');
    grid on;
    
    % Training loss (total)
    subplot(2,4,5);
    semilogy(1:length(losses), losses, 'b-', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('Total Loss');
    title('Multi-step GD Training Loss');
    grid on;
    
    % Multi-step losses breakdown (if available)
    if size(losses, 2) > 1
        subplot(2,4,6);
        step_losses = losses;  % This should be step_losses from the main code
        colors = {'b-', 'g-', 'm-', 'c-', 'y-'};
        for step = 1:min(5, size(step_losses, 2))
            semilogy(1:size(step_losses, 1), step_losses(:, step), colors{step}, ...
                'LineWidth', 1.5, 'DisplayName', sprintf('Step %d', step)); hold on;
        end
        xlabel('Iteration'); ylabel('Step Loss');
        title('Multi-step Loss Breakdown');
        legend('Location', 'best');
        grid on;
    end
    
    % Eigenvalue comparison
    subplot(2,4,7);
    plot(real(eigs_dmd), imag(eigs_dmd), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); hold on;
    plot(real(eigs_gd), imag(eigs_gd), 'bs', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    xlabel('Real Part'); ylabel('Imaginary Part');
    title('Eigenvalues (Continuous Time)');
    legend('DMD', 'Multi-step GD', 'Location', 'best');
    grid on; axis equal;
    
    % Add unit circle for stability reference
    theta = linspace(0, 2*pi, 100);
    plot(cos(theta), sin(theta), 'k--', 'LineWidth', 0.5);
    
    % Long-term error growth
    subplot(2,4,8);
    cumulative_error_dmd = cumsum(error_dmd.^2);
    cumulative_error_gd = cumsum(error_gd.^2);
    plot(t, cumulative_error_dmd, 'r-', 'LineWidth', 1.5); hold on;
    plot(t, cumulative_error_gd, 'b-', 'LineWidth', 1.5);
    xlabel('Time'); ylabel('Cumulative Error²');
    title('Long-term Error Accumulation');
    legend('DMD', 'Multi-step GD', 'Location', 'best');
    grid on;
    
    sgtitle('Van der Pol Koopman: DMD vs Multi-step GD with Random Fourier Features', 'FontSize', 14, 'FontWeight', 'bold');
end

%% Helper Functions