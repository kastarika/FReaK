%% Complete Example: Koopman DMD for Van der Pol Oscillator
% This script demonstrates how to use the identifyKoopmanDMD function
% with actual data from the Van der Pol oscillator

clear; clc; close all;

%% 1. Generate Data from Van der Pol Oscillator
fprintf('Generating data from Van der Pol oscillator...\n');

% Van der Pol dynamics
f_vdp = @(t,x) [x(2); (1-x(1)^2)*x(2)-x(1)];

% Simulation parameters
dt = 0.01;
t_span = 0:dt:50;
n_trajectories = 5;

% Generate multiple trajectories
trajectories = cell(n_trajectories, 1);
for i = 1:n_trajectories
    % Random initial conditions
    x0 = [3*rand-1.5; 3*rand-1.5];
    
    % Simulate
    [t, x] = ode45(f_vdp, t_span, x0);
    trajectories{i} = x;
    
    fprintf('  Trajectory %d: x0 = [%.2f, %.2f]\n', i, x0(1), x0(2));
end

%% 2. Create Observable Functions (Random Fourier Features)
fprintf('\nCreating observable functions...\n');

% Parameters for Random Fourier Features
n_features = 50;  % Number of random features
lengthscale = 1;   % RBF kernel lengthscale

% Generate random frequencies and phases
W = randn(n_features, 2) / lengthscale;
b = 2*pi*rand(n_features, 1);

% Define observable function
g = @(x) [x; sqrt(2)*cos(W*x + b)];
n_observables = n_features + 2;

fprintf('  Number of observables: %d\n', n_observables);

%% 3. Transform Data to Observable Space
fprintf('\nTransforming data to observable space...\n');

% Collect all data points
X_data = [];  % Observables at time k
Y_data = [];  % Observables at time k+1

for i = 1:n_trajectories
    x_traj = trajectories{i};
    
    % Transform each point and create pairs (x_k, x_{k+1})
    for j = 1:size(x_traj, 1)-1
        x_k = g(x_traj(j, :)');
        x_kp1 = g(x_traj(j+1, :)');
        
        X_data = [X_data, x_k];
        Y_data = [Y_data, x_kp1];
    end
end

fprintf('  Data matrix size: %d observables × %d samples\n', ...
    size(X_data, 1), size(X_data, 2));

%% 4. Identify Koopman Operator using Different Methods
fprintf('\n=== Identifying Koopman Operator ===\n');

methods = {'standard', 'exact', 'total', 'fbdmd'};
results = cell(length(methods), 1);

for m = 1:length(methods)
    fprintf('\n--- Method: %s ---\n', upper(methods{m}));
    
    % Identify Koopman operator
    [A, ~, info] = identifyKoopmanDMD(X_data, Y_data, [], ...
        'method', methods{m}, ...
        'rank', 'optimal', ...
        'threshold', 1e-10, ...
        'verbose', false);
    
    % Store results
    results{m}.A = A;
    results{m}.info = info;
    results{m}.method = methods{m};
    
    % Display metrics
    fprintf('  Rank used: %d\n', info.rank);
    fprintf('  Reconstruction error: %.2e\n', info.reconstruction_error);
    
    % Check stability (eigenvalues)
    max_eig = max(abs(info.eigenvalues));
    fprintf('  Max eigenvalue magnitude: %.4f\n', max_eig);
    if max_eig > 1.01
        fprintf('  WARNING: System may be unstable (max |λ| > 1)\n');
    end
end

%% 5. Test Predictions on New Initial Condition
fprintf('\n=== Testing Predictions ===\n');

% New test initial condition
x0_test = [2; 0];
fprintf('Test initial condition: [%.1f, %.1f]\n', x0_test(1), x0_test(2));

% True trajectory
[t_true, x_true] = ode45(f_vdp, 0:dt:50, x0_test);

% Predict using each Koopman model
figure('Position', [100, 100, 1200, 800]);

for m = 1:length(methods)
    A = results{m}.A;
    
    % Initial condition in observable space
    z0 = g(x0_test);
    
    % Predict
    n_steps = length(t_true);
    z_pred = zeros(n_observables, n_steps);
    z_pred(:, 1) = z0;
    
    for k = 2:n_steps
        z_pred(:, k) = A * z_pred(:, k-1);
    end
    
    % Extract original coordinates
    x_pred = z_pred(1:2, :)';
    
    % Plot
    subplot(2, 2, m);
    plot(x_true(:,1), x_true(:,2), 'w-', 'LineWidth', 2, 'DisplayName', 'True');
    hold on;
    plot(x_pred(:,1), x_pred(:,2), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Koopman');
    plot(x0_test(1), x0_test(2), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'DisplayName', 'Start');
    
    xlabel('x_1'); ylabel('x_2');
    title(sprintf('%s DMD (error: %.2e)', upper(methods{m}), results{m}.info.reconstruction_error));
    legend('Location', 'best');
    grid on;
    axis equal;
    xlim([-3, 3]); ylim([-3, 3]);
end

sgtitle('Koopman Model Predictions: Van der Pol Oscillator');

%% 6. Compare Time Series
figure('Position', [100, 100, 1200, 400]);

% Select best method (typically 'exact' or 'fbdmd')
best_method_idx = 2;  % exact DMD
A_best = results{best_method_idx}.A;

% Longer prediction
[t_long, x_long] = ode45(f_vdp, 0:dt:20, x0_test);
n_long = length(t_long);

% Koopman prediction
z_long = zeros(n_observables, n_long);
z_long(:, 1) = g(x0_test);
for k = 2:n_long
    z_long(:, k) = A_best * z_long(:, k-1);
end
x_koopman = z_long(1:2, :)';

% Plot time series
subplot(1, 2, 1);
plot(t_long, x_long(:,1), 'w-', 'LineWidth', 1.5, 'DisplayName', 'True x_1');
hold on;
plot(t_long, x_koopman(:,1), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Koopman x_1');
xlabel('Time'); ylabel('x_1');
title('First Component');
legend('Location', 'best');
grid on;

subplot(1, 2, 2);
plot(t_long, x_long(:,2), 'w-', 'LineWidth', 1.5, 'DisplayName', 'True x_2');
hold on;
plot(t_long, x_koopman(:,2), 'b--', 'LineWidth', 1.5, 'DisplayName', 'Koopman x_2');
xlabel('Time'); ylabel('x_2');
title('Second Component');
legend('Location', 'best');
grid on;

sgtitle(sprintf('Time Series Comparison (%s DMD)', upper(results{best_method_idx}.method)));

%% 7. Eigenvalue Analysis
figure('Position', [100, 100, 600, 600]);

% Plot eigenvalues for each method
colors = {'b', 'r', 'g', 'm'};
markers = {'o', 's', '^', 'd'};

% Unit circle
theta = linspace(0, 2*pi, 100);
plot(cos(theta), sin(theta), 'w--', 'LineWidth', 1, 'DisplayName', 'Unit Circle');
hold on;

for m = 1:length(methods)
    eigs = results{m}.info.eigenvalues;
    plot(real(eigs), imag(eigs), markers{m}, ...
        'Color', colors{m}, 'MarkerSize', 6, ...
        'DisplayName', sprintf('%s DMD', upper(methods{m})));
end

xlabel('Real'); ylabel('Imaginary');
title('Koopman Eigenvalues');
legend('Location', 'best');
grid on;
axis equal;
xlim([-1.5, 1.5]); ylim([-1.5, 1.5]);

%% 8. Summary Statistics
fprintf('\n=== Summary ===\n');
fprintf('Data generation:\n');
fprintf('  - %d trajectories\n', n_trajectories);
fprintf('  - %d time steps per trajectory\n', length(t_span));
fprintf('  - Total samples: %d\n', size(X_data, 2));
fprintf('\nObservable space:\n');
fprintf('  - Original dimension: 2\n');
fprintf('  - Lifted dimension: %d\n', n_observables);
fprintf('\nBest performing method: %s\n', upper(results{best_method_idx}.method));
fprintf('  - Reconstruction error: %.2e\n', results{best_method_idx}.info.reconstruction_error);
fprintf('  - Rank: %d\n', results{best_method_idx}.info.rank);

%% Helper Function - Include the identifyKoopmanDMD function here
% Note: In practice, you would save identifyKoopmanDMD as a separate .m file
% and ensure it's in your MATLAB path