function [u_opt, fval, exitflag] = optimize_koopman_gd(P, g, x0, phi_str, u0, lb, ub)
% optimize_koopman_gd
% Gradient-based falsification of Koopman system using STL robustness gradient
%
% Inputs:
%   P       - cell array of Koopman affine maps with fields A, B, c for each time step
%   g       - function handle for observable g(x)
%   x0      - initial state vector before applying g
%   phi_str - STL formula string (Breach format)
%   u0      - initial guess for input vector (all time steps concatenated)
%   lb, ub  - lower and upper bounds for input vector u
%
% Outputs:
%   u_opt   - optimized input vector minimizing STL robustness (falsification)
%   fval    - final objective value (robustness)
%   exitflag- exit flag from fmincon

    opts = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'iter', ...
                        'SpecifyObjectiveGradient', true, 'MaxIterations', 100);

    % Objective function wrapper that returns robustness and gradient
    fun = @(u) stl_cost_affine_koopman(u, P, g, x0, phi_str);

    % Run optimization
    [u_opt, fval, exitflag] = fmincon(fun, u0, [], [], [], [], lb, ub, [], opts);
end


function [cost, grad_u] = stl_cost_affine_koopman(u, P, g, x0, phi_str)
% Compute STL robustness and gradient w.r.t input u for affine Koopman system
%
% Inputs same as optimize_koopman_gd
%
% Outputs:
%   cost    - STL robustness (scalar)
%   grad_u  - gradient of robustness w.r.t u (vector)

    N = length(P);           % Number of time steps
    m = size(P{1}.B, 2);    % Number of inputs per step

    % Reshape input vector u to matrix (m x N)
    u_mat = reshape(u, m, N);

    % Precompute g(x0)
    gx0 = g(x0);
    dim_g = length(gx0);

    % Propagate states using Koopman affine form
    % x(t_i) = A_i * g(x0) + B_i * u(:,i) + c_i
    states = zeros(length(gx0), N);
    for i = 1:N
        states(:, i) = P{i}.A * gx0 + P{i}.B * u_mat(:, i); %+ P{i}.c;
    end

    % --- Wrap states into struct with x1..x10 fields ---
    x_struct = struct();
    max_vars = min(dim_g, 10);  % support up to 10 vars
    for j = 1:max_vars
        fieldname = sprintf('x%d', j);
        x_struct.(fieldname) = states(j, :);
    end

    % Compute STL robustness and gradient w.r.t. states (using your STL gradient function)
    [robustness, grad_struct] = eval_stl_recursive(phi_str, x_struct);
    
    % --- Convert grad_struct back into grad_states matrix ---
    grad_states = zeros(dim_g, N);
    for j = 1:max_vars
        fieldname = sprintf('x%d', j);
        grad_states(j, :) = grad_struct.(fieldname);
    end

    
    % grad_states is (dim_g x N), robustness gradient w.r.t. states

    % Chain rule: grad robustness w.r.t u
    % grad_u_i = B_i' * grad_states(:,i)
    % BUT states depend on g(x0), which depends on x0; we optimize w.r.t u only here
    grad_u_mat = zeros(m, N);
    for i = 1:N
        grad_u_mat(:, i) = P{i}.B' * grad_states(:, i);
    end

    % Vectorize grad_u_mat to grad_u (column vector)
    grad_u = grad_u_mat(:);

    % Return robustness cost and gradient
    cost = robustness;
end

