% %% Example test for optimize_koopman_gd
% 
% % Horizon length
% N = 5;       % number of steps
% m = 1;       % input dimension
% n = 1;       % state dimension
% dim_g = 1;   % observable dimension (same as state here)
% 
% % Define Koopman affine maps P{i}: x_{i+1} = A*g(x0) + B*u + c
% P = cell(1, N);
% for i = 1:N
%     P{i}.A = 0.9;                  % Koopman dynamics (decaying memory)
%     P{i}.B = 1.0;                  % input influence
%     P{i}.c = 0.1;                  % small bias
% end
% 
% % Observable function g(x) = x (identity here)
% g = @(x) x;
% 
% % Initial condition
% x0 = 0.5;
% 
% % STL formula string (Breach format, toy example)
% % Example: "alw_[0,5] (x < 1.0)"  meaning "always within horizon state < 1.0"
% phi_str = 'alw_[0,5] (x1[t] < 1.0)';
% 
% % Initial guess for inputs (flattened vector of length m*N)
% u0 = 0.5 * ones(m*N, 1);
% 
% % Lower/upper bounds
% lb = -2 * ones(m*N, 1);
% ub =  2 * ones(m*N, 1);
% 
% % Run the optimization
% [u_opt, fval, exitflag] = optimize_koopman_gd(P, g, x0, phi_str, u0, lb, ub);
% 
% disp('Optimized input:');
% disp(u_opt);
% disp('Final robustness value:');
% disp(fval);
% disp('Exit flag:');
% disp(exitflag);
% 
% 

%% test_koopman_falsification.m
% Test script for your optimize_koopman_gd function

% clear; clc;

% Load Koopman model
data = load('testkoopModel.mat');  % expects testkoopModel.mat
koopModel = data.koopModel;        % the struct inside the file

% Extract model components
A = koopModel.A;      % dim_z x dim_z
B = koopModel.B;      % dim_z x dim_u
g = koopModel.g;      % function handle
x0 = koopModel.x0;    % dim_x x 1
N = koopModel.N;
u0 = koopModel.U;


dim_z = length(g(x0));
dim_u = size(B,2);
% N = 10;                % number of time steps
% u0 = zeros(dim_u*N,1); % initial guess for input sequence


% STL formula (example)
phi_str = 'always(x1 < 5)';  % replace with your STL requirement

% Input bounds (example)
lb = zeros(dim_u*N,1);
ub = ones(dim_u*N,1);

% Run gradient-based falsification
[u_opt, fval, exitflag] = optimize_koopman_gd_3(A, B, g, x0, phi_str, u0, lb, ub, N);

% Display results
disp('Optimized input sequence:');
disp(u_opt);

disp('pre input')
disp(u0);

disp('Final STL robustness:');
disp(fval);

disp('Exit flag:');
disp(exitflag);

