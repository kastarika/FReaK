%% Simple Example: Using identifyKoopmanGradient for Van der Pol Oscillator
% This example shows how to call the identifyKoopmanGradient function
% Assumes identifyKoopmanGradient.m is in your MATLAB path

clear; clc; close all;

%% 1. Setup System and Observable Functions
% Van der Pol oscillator
f_vdp = @(t,x) [x(2); (1-x(1)^2)*x(2)-x(1)];

% Create Random Fourier Features for observables
n_features = 50;
lengthscale = 1;
W = randn(n_features, 2) / lengthscale;
b = 2*pi*rand(n_features, 1);
g = @(x) [x; sqrt(2)*cos(W*x + b)];
g_inv = @(z) z(1:2);  % Extract original states

n_obs = n_features + 2;

%% 2. Generate Training Trajectories in Observable Space
dt = 0.1;
t_span = 0:dt:20;
n_trajectories = 5;

% Cell array to store trajectories
X_traj = cell(n_trajectories, 1);

fprintf('Generating %d training trajectories...\n', n_trajectories);
for i = 1:n_trajectories
    % Random initial condition
    x0 = [3*rand-1.5; 3*rand-1.5];
    
    % Simulate system
    [~, x] = ode45(f_vdp, t_span, x0);
    
    % Transform to observable space
    X_obs = zeros(n_obs, length(t_span));
    for j = 1:length(t_span)
        X_obs(:, j) = g(x(j,:)');
    end
    
    X_traj{i} = X_obs;
end

%% 3. Train Koopman Operator using Gradient Descent
fprintf('\nTraining Koopman operator with gradient descent...\n');

% Call the gradient descent function
[A_koopman, ~, history] = identifyKoopmanGradient(X_traj, [], ...
    'n_steps', 100,    ...          % Optimize over 15-step predictions
    'learning_rate', 0.01, ...    % Initial learning rate
    'max_iter', 1000,    ...       % Maximum iterations
    'optimizer', 'adam',...        % Use Adam optimizer
    'init_method', 'dmd',...       % Initialize with DMD
    'regularization', 1e-5, ...    % L2 regularization
    'lr_decay', 0.995,   ...       % Learning rate decay
    'verbose', true,   ...         % Show progress
    'plot_loss', true);         % Plot training loss

%% 4. Test the Learned Koopman Operator
fprintf('\nTesting on new trajectory...\n');

% New test initial condition
x0_test = [2; 0];
t_test = 0:dt:20;

% True trajectory
[~, x_true] = ode45(f_vdp, t_test, x0_test);

% Koopman prediction
z0 = g(x0_test);
z_pred = zeros(n_obs, length(t_test));
z_pred(:, 1) = z0;

for k = 2:length(t_test)
    z_pred(:, k) = A_koopman * z_pred(:, k-1);
end

disp(z_pred(1:2,:))
disp(size(z_pred))

% Extract original coordinates
x_pred = g_inv(z_pred);
disp(x_pred)
disp(size(x_pred))

%% 5. Visualize Results
figure('Position', [100, 100, 1200, 500]);

% Phase portrait
subplot(1, 2, 1);
plot(x_true(:,1), x_true(:,2), 'k-', 'LineWidth', 2, 'DisplayName', 'True');
hold on;
plot(x_pred(:,1), x_pred(:,2), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Koopman');
plot(x0_test(1), x0_test(2), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'DisplayName', 'Start');
xlabel('x_1'); ylabel('x_2');
title('Phase Portrait: 20 second prediction');
legend('Location', 'best');
grid on;
axis equal;
xlim([-3, 3]); ylim([-3, 3]);

% Time series
subplot(1, 2, 2);
plot(t_test, x_true(:,1), 'k-', 'LineWidth', 1.5, 'DisplayName', 'True x_1');
hold on;
plot(t_test, x_pred(:,1), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Predicted x_1');
xlabel('Time (s)'); ylabel('x_1');
title('Time Series Comparison');
legend('Location', 'best');
grid on;
xlim([0, 20]);

sgtitle('Koopman Model Learned via Gradient Descent');

% Compute and display error
error = vecnorm(x_pred - x_true, 2, 2);
fprintf('\nPrediction error statistics:\n');
fprintf('  Mean error: %.4f\n', mean(error));
fprintf('  Max error: %.4f\n', max(error));
fprintf('  Final error: %.4f\n', error(end));

%% 6. Compare Different Hyperparameters (Optional)
fprintf('\n=== Testing different multi-step horizons ===\n');

horizons = [1, 5, 10, 20, 30];
final_losses = zeros(length(horizons), 1);

for i = 1:length(horizons)
    fprintf('Training with horizon = %d...\n', horizons(i));
    
    [~, ~, hist] = identifyKoopmanGradient(X_traj, [], ...
        'n_steps', horizons(i), ...
        'learning_rate', 0.001, ...
        'max_iter', 500, ...
        'optimizer', 'adam', ...
        'init_method', 'dmd', ...
        'regularization', 1e-5, ...
        'verbose', false, ...
        'plot_loss', false);
    
    final_losses(i) = hist.loss(end);
    fprintf('  Final loss: %.6e\n', final_losses(i));
end

figure;
plot(horizons, final_losses, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('Multi-step Horizon');
ylabel('Final Training Loss');
title('Effect of Multi-step Horizon on Training Loss');
grid on;