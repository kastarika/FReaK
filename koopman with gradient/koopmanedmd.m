%% edmd_fit_and_sim.m
% Usage examples at the bottom.

function koopman = edmd_fit(X, Y, g, U, reg)
% EDMD fit
% X : n x m    matrix of states at times k (each column a state)
% Y : n x m    matrix of states at times k+1 (same number of columns)
% g : function handle -> z = g(X) returns N x m lifted observables
% U : p x m    (optional) input sequence applied between X->Y (same m). If empty, no inputs.
% reg: scalar  Tikhonov regularization (optional, default 1e-8)
%
% Returns struct koopman with fields:
%   koopman.A (N x N) -- Koopman operator on lifted states
%   koopman.B (N x p) -- input matrix (empty if no U provided)
%   koopman.C (n x N) -- reconstruction matrix (x ≈ C*z)
%   koopman.g (function handle) -- the lifting function used
%   koopman.Z, koopman.Zp (lifted snapshots) -- optional for debugging

    if nargin < 4
        U = [];
    end
    if nargin < 5 || isempty(reg)
        reg = 1e-8;
    end

    % basic checks
    [n, m] = size(X);
    [ny, my] = size(Y);
    assert(n == ny, 'X and Y must have same number of rows (state dimension).');
    assert(m == my, 'X and Y must have same number of columns (snapshot count).');

    % lift snapshots
    Z  = g(X);   % N x m
    Zp = g(Y);   % N x m
    [N, mZ] = size(Z);
    assert(mZ == m, 'g must return N x m lifted snapshots.');

    % Solve for A (and B if inputs present)
    if isempty(U)
        % Zp = A * Z
        % Solve A = Zp * pinv(Z) but use ridge for numerical stability:
        % A = Zp * Z' * inv(Z*Z' + reg*I)
        SzzT = (Z * Z');              % N x N
        A = (Zp * Z') / (SzzT + reg*eye(N));
        B = [];
    else
        [p, mU] = size(U);
        assert(mU == m, 'U must have same number of snapshots (columns) as X,Y.');
        % Stack [Z; U] of size (N+p) x m
        stacked = [Z; U];  % (N+p) x m
        S = stacked * stacked';            % (N+p) x (N+p)
        Theta = (Zp * stacked') / (S + reg*eye(size(S))); % (N) x (N+p)
        A = Theta(:, 1:N);
        B = Theta(:, N+1:end);
    end

    % Reconstruction matrix C: X ≈ C * Z  -> C = X * pinv(Z)
    % Use ridge: C = X * Z' * inv(Z * Z' + reg*I)
    C = (X * Z') / (Z * Z' + reg*eye(N));

    % Return struct
    koopman.A = A;
    koopman.B = B;
    koopman.C = C;
    koopman.g = g;
    koopman.Z = Z;
    koopman.Zp = Zp;
end

%% Simulation helper
function [Xsim, Zsim] = simulate_koopman(koopman, x0, Useq, steps)
% Simulate Koopman model forward for 'steps' steps
% koopman: struct from edmd_fit
% x0: n x 1 initial state
% Useq: p x steps input sequence (optional - if koopman.B is empty, ignored)
% steps: integer number of time steps to simulate (if empty, will use columns of Useq)
%
% Returns:
%   Xsim: n x (steps+1) states including x0
%   Zsim: N x (steps+1) lifted states including z0

    if nargin < 4 || isempty(steps)
        if ~isempty(Useq)
            steps = size(Useq, 2);
        else
            error('Either provide steps or Useq with enough columns.');
        end
    end

    A = koopman.A;
    B = koopman.B;
    C = koopman.C;
    g = koopman.g;

    z0 = g(x0);           % N x 1
    N = size(z0,1);
    n = size(C,1);

    Zsim = zeros(N, steps+1);
    Xsim = zeros(n, steps+1);

    Zsim(:,1) = z0;
    Xsim(:,1) = x0;

    for k = 1:steps
        uk = [];
        if ~isempty(Useq)
            uk = Useq(:,k);
        end

        if isempty(B)
            Znext = A * Zsim(:,k);
        else
            Znext = A * Zsim(:,k) + B * uk;
        end
        Zsim(:,k+1) = Znext;
        Xsim(:,k+1) = C * Znext;
    end
end

%% Example usage
% The following example demonstrates usage with polynomial observables up to degree 2.
% Uncomment and run as a script to test.

%{
% -- create synthetic linear dynamical system for demo
n = 2;   % state dim
A_true = [0.9, 0.2; -0.1, 0.95];
B_true = [0.1; 0.05];
T = 200;
X = zeros(n, T);
U = zeros(1, T);
X(:,1) = [1; 0.5];
for k = 1:T-1
    U(:,k) = 0.5*sin(2*pi*0.01*k);
    X(:,k+1) = A_true*X(:,k) + B_true*U(:,k);
end
% drop last to form pairs
Xdata = X(:,1:end-1);
Ydata = X(:,2:end);
Udata = U(:,1:end-1);

% -- define lifting function g (polynomial up to degree 2)
g = @(Xmat) poly2_obs(Xmat);

% -- fit EDMD with inputs
koop = edmd_fit(Xdata, Ydata, g, Udata, 1e-8);

% -- simulate from x0
x0 = X(:,1);
[XSIM, ZSIM] = simulate_koopman(koop, x0, Udata(:,1:50), 50);

% plot first state
figure;
plot(0:50, XSIM(1,:), '-o'); hold on;
plot(0:50, X(1,1:51), '--x'); legend('EDMD predicted x1','True x1'); grid on;
%}

% end

%% helper: polynomial observables degree 2
function Z = poly2_obs(X)
% X: n x m
% Z: (n + n*(n+1)/2) x m  -> [x; pairwise products x_i*x_j] (including squares)
    [n, m] = size(X);
    % first the original states
    Z = X;
    % then pairwise products (i <= j)
    for i = 1:n
        for j = i:n
            prodij = X(i,:) .* X(j,:);
            Z = [Z; prodij];
        end
    end
end
