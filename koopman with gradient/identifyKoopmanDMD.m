function [A, B, sys_info] = identifyKoopmanDMD(X, Y, U, varargin)
% IDENTIFYKOOPMANDMD Identifies a Koopman operator using Dynamic Mode Decomposition
%
% Syntax:
%   A = identifyKoopmanDMD(X, Y)
%   A = identifyKoopmanDMD(X, Y, U)
%   [A, B, sys_info] = identifyKoopmanDMD(X, Y, U, 'OptionName', OptionValue, ...)
%
% Description:
%   Computes the Koopman operator that best approximates the dynamics
%   Y = A*X + B*U using Dynamic Mode Decomposition (DMD) or its variants.
%
% Inputs:
%   X - [n x m] matrix of observables at time k (n features, m samples)
%   Y - [n x m] matrix of observables at time k+1
%   U - [p x m] matrix of control inputs (optional, use [] for autonomous)
%
% Options (Name-Value pairs):
%   'method' - DMD variant to use:
%       'standard' - Standard DMD (default)
%       'exact'    - Exact DMD using SVD
%       'total'    - Total least squares DMD
%       'fbdmd'    - Forward-backward DMD (more robust)
%   'rank' - Truncation rank for SVD (integer or 'optimal')
%           'optimal' uses automatic rank selection (default)
%   'threshold' - Singular value threshold for rank truncation (default: 1e-10)
%   'tikhonov' - Tikhonov regularization parameter (default: 0)
%   'dt' - Time step for continuous-time conversion (default: [] for discrete)
%   'verbose' - Display progress information (default: false)
%
% Outputs:
%   A - [n x n] Koopman operator matrix
%   B - [n x p] Input matrix (empty if no inputs)
%   sys_info - Structure with additional information:
%       .singular_values - Singular values from decomposition
%       .rank - Rank used for truncation
%       .reconstruction_error - Reconstruction error ||Y - A*X - B*U||_F
%       .modes - Koopman modes
%       .eigenvalues - Koopman eigenvalues
%       .method - Method used
%
% Example:
%   % Generate data from Van der Pol oscillator
%   dt = 0.01; t = 0:dt:10;
%   [t, x] = ode45(@(t,x) [x(2); (1-x(1)^2)*x(2)-x(1)], t, [2; 0]);
%   
%   % Create observables (Random Fourier Features)
%   n_obs = 100;
%   W = randn(n_obs, 2);
%   b = 2*pi*rand(n_obs, 1);
%   g = @(x) [x; sqrt(2)*cos(W*x + b)];
%   
%   % Transform data
%   X = g(x(1:end-1,:)');
%   Y = g(x(2:end,:)');
%   
%   % Identify Koopman operator
%   [A, ~, info] = identifyKoopmanDMD(X, Y, [], 'method', 'exact');
%
% References:
%   [1] Tu et al., "On Dynamic Mode Decomposition: Theory and Applications", 2014
%   [2] Kutz et al., "Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems", 2016
%   [3] Williams et al., "A Data-Driven Approximation of the Koopman Operator", 2015

% Parse inputs
p = inputParser;
addRequired(p, 'X', @(x) isnumeric(x) && ismatrix(x));
addRequired(p, 'Y', @(x) isnumeric(x) && ismatrix(x));
addOptional(p, 'U', [], @(x) isnumeric(x) || isempty(x));
addParameter(p, 'method', 'standard', @(x) any(strcmpi(x, {'standard', 'exact', 'total', 'fbdmd'})));
addParameter(p, 'rank', 'optimal', @(x) (isnumeric(x) && x > 0) || strcmpi(x, 'optimal'));
addParameter(p, 'threshold', 1e-10, @(x) isnumeric(x) && x > 0);
addParameter(p, 'tikhonov', 0, @(x) isnumeric(x) && x >= 0);
addParameter(p, 'dt', [], @(x) isnumeric(x) || isempty(x));
addParameter(p, 'verbose', false, @islogical);

parse(p, X, Y, varargin{:});
opts = p.Results;
U = p.Results.U;

% Validate dimensions
[n, m] = size(X);
assert(all(size(Y) == [n, m]), 'X and Y must have the same dimensions');

if ~isempty(U)
    assert(size(U, 2) == m, 'U must have the same number of samples as X and Y');
end

% Display info
if opts.verbose
    fprintf('=== Koopman DMD Identification ===\n');
    fprintf('Data: %d observables, %d samples\n', n, m);
    fprintf('Method: %s\n', opts.method);
    if ~isempty(U)
        fprintf('Inputs: %d dimensions\n', size(U, 1));
    end
end

% Select method
switch lower(opts.method)
    case 'standard'
        [A, B, info] = dmd_standard(X, Y, U, opts);
    case 'exact'
        [A, B, info] = dmd_exact(X, Y, U, opts);
    case 'total'
        [A, B, info] = dmd_total(X, Y, U, opts);
    case 'fbdmd'
        [A, B, info] = dmd_fbdmd(X, Y, U, opts);
    otherwise
        error('Unknown method: %s', opts.method);
end

% Add method info
info.method = opts.method;

% Convert to continuous time if requested
if ~isempty(opts.dt)
    if opts.verbose
        fprintf('Converting to continuous time (dt = %g)\n', opts.dt);
    end
    A = logm(A) / opts.dt;
    % Note: B remains the same for small dt
end

% Compute eigendecomposition
[V, D] = eig(A);
info.modes = V;
info.eigenvalues = diag(D);

% Compute reconstruction error
if ~isempty(U)
    Y_pred = A * X + B * U;
else
    Y_pred = A * X;
end
info.reconstruction_error = norm(Y - Y_pred, 'fro') / norm(Y, 'fro');

if opts.verbose
    fprintf('Rank used: %d\n', info.rank);
    fprintf('Relative reconstruction error: %.2e\n', info.reconstruction_error);
    fprintf('==================================\n');
end

% Store system info
sys_info = info;

end

%% Standard DMD
function [A, B, info] = dmd_standard(X, Y, U, opts)
% Standard DMD: directly solve Y = A*X in least squares sense

[n, m] = size(X);

% Augment with inputs if present
if ~isempty(U)
    X_aug = [X; U];
else
    X_aug = X;
end

% Solve least squares problem with optional Tikhonov regularization
if opts.tikhonov > 0
    AB = Y * X_aug' / (X_aug * X_aug' + opts.tikhonov * eye(size(X_aug, 1)));
else
    AB = Y / X_aug;  % MATLAB's backslash for numerical stability
end

% Extract A and B
A = AB(:, 1:n);
if ~isempty(U)
    B = AB(:, n+1:end);
else
    B = [];
end

% No SVD info for standard method
info.singular_values = [];
info.rank = n;

end

%% Exact DMD
function [A, B, info] = dmd_exact(X, Y, U, opts)
% Exact DMD using SVD for numerical stability

[n, m] = size(X);

% Augment with inputs
if ~isempty(U)
    X_aug = [X; U];
else
    X_aug = X;
end

% SVD of augmented data matrix
[Ux, Sx, Vx] = svd(X_aug, 'econ');
singular_values = diag(Sx);

% Determine rank
if strcmpi(opts.rank, 'optimal')
    % Automatic rank selection based on threshold
    rank = sum(singular_values > opts.threshold * max(singular_values));
else
    rank = min(opts.rank, length(singular_values));
end

% Truncate
Ux = Ux(:, 1:rank);
Sx = Sx(1:rank, 1:rank);
Vx = Vx(:, 1:rank);

% Compute reduced operator
if opts.tikhonov > 0
    % With regularization
    Sx_inv = diag(singular_values(1:rank) ./ (singular_values(1:rank).^2 + opts.tikhonov));
    Atilde = Y * Vx * Sx_inv * Ux';
else
    % Without regularization
    Atilde = Y * Vx / Sx * Ux';
end

% Extract A and B
A = Atilde(:, 1:n);
if ~isempty(U)
    B = Atilde(:, n+1:end);
else
    B = [];
end

% Store info
info.singular_values = singular_values;
info.rank = rank;

end

%% Total Least Squares DMD
function [A, B, info] = dmd_total(X, Y, U, opts)
% Total least squares DMD - accounts for noise in both X and Y

[n, m] = size(X);

% Augment with inputs
if ~isempty(U)
    X_aug = [X; U];
    p = size(U, 1);
else
    X_aug = X;
    p = 0;
end

% Form augmented matrix
Z = [X_aug; Y];

% SVD of augmented matrix
[Uz, Sz, Vz] = svd(Z, 'econ');
singular_values = diag(Sz);

% Determine rank
if strcmpi(opts.rank, 'optimal')
    rank = sum(singular_values > opts.threshold * max(singular_values));
else
    rank = min(opts.rank, length(singular_values));
end

% Truncate
Uz = Uz(:, 1:rank);
Sz = Sz(1:rank, 1:rank);
Vz = Vz(:, 1:rank);

% Split the truncated U matrix
Ux = Uz(1:n+p, :);
Uy = Uz(n+p+1:end, :);

% Compute the operator
if rcond(Ux' * Ux) < 1e-12
    warning('Nearly singular matrix in total least squares DMD');
    % Fall back to standard pseudoinverse
    AB = Uy * pinv(Ux);
else
    AB = Uy / Ux;
end

% Extract A and B
A = AB(:, 1:n);
if ~isempty(U)
    B = AB(:, n+1:end);
else
    B = [];
end

% Store info
info.singular_values = singular_values;
info.rank = rank;

end

%% Forward-Backward DMD
function [A, B, info] = dmd_fbdmd(X, Y, U, opts)
% Forward-backward DMD for improved accuracy
% Combines forward (X->Y) and backward (Y->X) DMD

[n, m] = size(X);

% Forward DMD
[A_fwd, B_fwd, info_fwd] = dmd_exact(X, Y, U, opts);

% Backward DMD (only for autonomous case)
if isempty(U)
    [A_bwd, ~, info_bwd] = dmd_exact(Y, X, [], opts);
    
    % Combine forward and backward
    % Method 1: Average in log space (for stability)
    if rcond(A_bwd) > 1e-12
        A = real(expm((logm(A_fwd) - logm(inv(A_bwd))) / 2));
    else
        % Fall back to forward only if backward is singular
        A = A_fwd;
    end
    
    B = [];
    
    % Average singular values info
    info.singular_values = (info_fwd.singular_values + info_bwd.singular_values) / 2;
    info.rank = info_fwd.rank;
else
    % With control inputs, only use forward DMD
    A = A_fwd;
    B = B_fwd;
    info = info_fwd;
end

end

%% Example usage function
function example_koopman_dmd()
% Example: Van der Pol oscillator with Koopman DMD

% Generate data
dt = 0.01;
t = 0:dt:20;
f_vdp = @(t,x) [x(2); (1-x(1)^2)*x(2)-x(1)];

% Multiple trajectories with different initial conditions
X_data = [];
Y_data = [];

for ic = 1:5
    x0 = [3*rand-1.5; 3*rand-1.5];
    [~, x] = ode45(f_vdp, t, x0);
    
    % Create Random Fourier Feature observables
    if ic == 1
        n_features = 100;
        W = randn(n_features, 2) / 1;  % lengthscale = 1
        b = 2*pi*rand(n_features, 1);
        g = @(x) [x; sqrt(2)*cos(W*x + b)];
    end
    
    % Transform and collect data
    for j = 1:length(t)-1
        X_data = [X_data, g(x(j,:)')];
        Y_data = [Y_data, g(x(j+1,:)')];
    end
end

% Identify Koopman operator
fprintf('\n=== Van der Pol Koopman Identification ===\n');

% Try different methods
methods = {'standard', 'exact', 'fbdmd'};
for i = 1:length(methods)
    fprintf('\nMethod: %s\n', methods{i});
    [A, ~, info] = identifyKoopmanDMD(X_data, Y_data, [], ...
        'method', methods{i}, ...
        'rank', 'optimal', ...
        'verbose', true);
    
    % Test prediction
    x_test = [1; 0];
    x_lifted = g(x_test);
    
    % Simulate for a short time
    n_steps = 100;
    x_pred = zeros(n_features + 2, n_steps);
    x_pred(:,1) = x_lifted;
    for k = 2:n_steps
        x_pred(:,k) = A * x_pred(:,k-1);
    end
    
    % Extract original coordinates
    x_orig = x_pred(1:2, :);
    
    fprintf('Eigenvalues (magnitude): min=%.4f, max=%.4f\n', ...
        min(abs(info.eigenvalues)), max(abs(info.eigenvalues)));
end

end