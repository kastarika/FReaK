function [A, B, history] = identifyKoopmanGradient(X_traj, U_traj, varargin)
% IDENTIFYKOOPMANGRADIENT Identifies Koopman operator using gradient descent
% with multi-step prediction loss
%
% Syntax:
%   A = identifyKoopmanGradient(X_traj)
%   A = identifyKoopmanGradient(X_traj, U_traj)
%   [A, B, history] = identifyKoopmanGradient(X_traj, U_traj, 'Option', value, ...)
%
% Inputs:
%   X_traj - Cell array of trajectories, each [n x T] (n observables, T time steps)
%   U_traj - Cell array of control inputs (optional), each [m x T-1]
%
% Options:
%   'n_steps' - Number of future steps for loss (default: 10)
%   'learning_rate' - Initial learning rate (default: 0.01)
%   'max_iter' - Maximum iterations (default: 1000)
%   'tol' - Convergence tolerance (default: 1e-6)
%   'batch_size' - Batch size for mini-batch gradient descent (default: all)
%   'momentum' - Momentum coefficient (default: 0.9)
%   'regularization' - L2 regularization parameter (default: 1e-4)
%   'init_method' - Initialization: 'dmd', 'random', 'identity' (default: 'dmd')
%   'optimizer' - 'sgd', 'adam', 'rmsprop' (default: 'adam')
%   'lr_decay' - Learning rate decay factor (default: 0.99)
%   'verbose' - Display progress (default: true)
%   'plot_loss' - Plot loss during training (default: true)
%
% Outputs:
%   A - [n x n] Koopman operator
%   B - [n x m] Input matrix (empty if no inputs)
%   history - Structure with training history
%
% Example:
%   % Generate trajectories in observable space
%   X_traj = cell(10, 1);
%   for i = 1:10
%       X_traj{i} = randn(100, 50); % 100 observables, 50 time steps
%   end
%   [A, ~, history] = identifyKoopmanGradient(X_traj, [], 'n_steps', 20);

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

if opts.verbose
    fprintf('=== Koopman Gradient Descent ===\n');
    fprintf('Trajectories: %d\n', n_traj);
    fprintf('Observables: %d\n', n);
    if has_inputs
        fprintf('Inputs: %d\n', m);
    end
    fprintf('Multi-step horizon: %d\n', opts.n_steps);
    fprintf('Optimizer: %s\n', opts.optimizer);
    fprintf('================================\n');
end

% Initialize A and B
[A, B] = initialize_matrices(X_traj, U_traj, n, m, opts.init_method);

% Prepare for optimization
history.loss = [];
history.grad_norm = [];
history.learning_rate = [];

% Initialize optimizer state
opt_state = init_optimizer(opts.optimizer, n, m);

% Setup figure for live plotting
if opts.plot_loss
    fig = figure('Name', 'Koopman Training Progress');
    subplot(2,1,1);
    h_loss = animatedline('Color', 'b', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('Loss');
    title('Multi-step Prediction Loss');
    grid on;
    
    subplot(2,1,2);
    h_grad = animatedline('Color', 'r', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('Gradient Norm');
    title('Gradient Norm');
    grid on;
    set(gca, 'YScale', 'log');
end

% Main optimization loop
lr = opts.learning_rate;
best_loss = inf;
best_A = A;
best_B = B;
patience = 50;
no_improve_count = 0;

for iter = 1:opts.max_iter
    % disp('aksdjflkasjdflkjasdlkfjk')
    % Get batch of trajectories
    disp(iter)
    if opts.batch_size < n_traj
        batch_idx = randperm(n_traj, min(opts.batch_size, n_traj));
    else
        batch_idx = 1:n_traj;
    end
    
    % Compute loss and gradients
    if has_inputs
        U_batch = U_traj(batch_idx);
    else
        U_batch = [];
    end
    % disp('asdjflkasjdfkl')
    [loss, grad_A, grad_B] = compute_multistep_loss_and_gradient(...
        A, B, X_traj(batch_idx), U_batch, opts.n_steps, opts.regularization);
    % disp('asdfasdfasdf')
    % Update parameters using chosen optimizer
    [A, B, opt_state] = update_parameters(...
        A, B, grad_A, grad_B, lr, opts.optimizer, opt_state, opts.momentum);
    % disp('asdfjlkasjdflkjaldksfj')
    % Store history
    history.loss(iter) = loss;
    history.grad_norm(iter) = norm([grad_A(:); grad_B(:)]);
    history.learning_rate(iter) = lr;
    % disp('asdjflasjdflkjasldk')
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
    
    % Update plots
    if opts.plot_loss && mod(iter, 2) == 0
        addpoints(h_loss, iter, loss);
        addpoints(h_grad, iter, history.grad_norm(iter));
        drawnow;
    end
    
    % Display progress
    if opts.verbose && mod(iter, 100) == 0
        fprintf('Iter %4d: Loss = %.6e, |∇| = %.6e, LR = %.6e\n', ...
            iter, loss, history.grad_norm(iter), lr);
    end
    
    % Check convergence
    if iter > 10 && abs(history.loss(iter) - history.loss(iter-1)) < opts.tol
        if opts.verbose
            fprintf('Converged at iteration %d\n', iter);
        end
        break;
    end
    
    % Early stopping
    if no_improve_count > patience
        if opts.verbose
            fprintf('Early stopping at iteration %d\n', iter);
        end
        break;
    end
end

% Return best model
A = best_A;
B = best_B;

if opts.verbose
    fprintf('Final loss: %.6e\n', best_loss);
    fprintf('================================\n');
end

end

%% Helper Functions

function [A, B] = initialize_matrices(X_traj, U_traj, n, m, method)
% Initialize A and B matrices

switch lower(method)
    case 'dmd'
        % Use DMD for initialization
        X = [];
        Y = [];
        
        for i = 1:length(X_traj)
            X = [X, X_traj{i}(:, 1:end-1)];
            Y = [Y, X_traj{i}(:, 2:end)];
        end
        
        if m == 0 || isempty(U_traj)
            % No inputs - autonomous system
            A = Y / X;
            B = zeros(n, 0);
        else
            % With inputs
            U = [];
            for i = 1:length(U_traj)
                U = [U, U_traj{i}];
            end
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

function [loss, grad_A, grad_B] = compute_multistep_loss_and_gradient(A, B, X_traj, U_traj, n_steps, lambda)
% Compute multi-step prediction loss and gradients

n = size(A, 1);
m = size(B, 2);
n_traj = length(X_traj);
has_inputs = ~isempty(U_traj) && m > 0;

loss = 0;
grad_A = zeros(n, n);
grad_B = zeros(n, m);

% For each trajectory
for traj_idx = 1:n_traj
    X = X_traj{traj_idx};
    T = size(X, 2);
    
    if has_inputs
        U = U_traj{traj_idx};
    end
    
    % For each starting point in the trajectory
    n_steps= T - 2;
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
n_samples = n_traj * (T - n_steps) * n_steps;
loss = loss / n_traj;
grad_A = grad_A / n_samples;
grad_B = grad_B / n_samples;

% Add L2 regularization
if lambda > 0
    loss = loss + 0.5 * lambda * (norm(A, 'fro')^2 + norm(B, 'fro')^2);
    grad_A = grad_A + lambda * A;
    grad_B = grad_B + lambda * B;
end

end

function opt_state = init_optimizer(optimizer, n, m)
% Initialize optimizer state

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

function [A, B, opt_state] = update_parameters(A, B, grad_A, grad_B, lr, optimizer, opt_state, momentum)
% Update parameters using specified optimizer

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