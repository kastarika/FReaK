function [u_opt, fval, exitflag] = koopman_gd(A, B, g, x0, phi_str, u0, lb, ub, N)
    % Dimensions
    dim_x = length(x0);
    dim_u = size(B,2);
    x0 = x0(:);
    % disp(x0);
    % disp(size(x0));
    z0 = g(x0);
    dim_z = length(z0);

    % Projection matrix
    C = [eye(dim_x), zeros(dim_x, dim_z - dim_x)];

    % Precompute A^(k-1)*B 
    AB_cells = cell(N,1);
    A_power = eye(dim_z);
    for k = 1:N
        AB_cells{k} = A_power * B;
        % disp(AB_cells{k})
        % keyboard;
        A_power = A * A_power;  % prepare for next power
    end

    lb = repmat(lb, N, 1);   % (10x1)
    ub = repmat(ub, N, 1);   % (10x1)

    % fmincon options
    opts = optimoptions('fmincon','Algorithm','sqp','Display','iter', ...
                        'SpecifyObjectiveGradient', true, 'MaxIterations', 100);
    % opts = optimoptions('fmincon','Algorithm','sqp','Display','iter', ...
    % 'SpecifyObjectiveGradient', true, 'MaxIterations', 100, ...
    % 'OutputFcn', @stop_when_negative);

    % Objective wrapper
    fun = @(u) stl_cost_devec_pre(u, A, B, g, z0, phi_str, C, dim_u, N, AB_cells);

    % Run optimization
    [u_opt, fval, exitflag] = fmincon(fun, u0, [], [], [], [], lb, ub, [], opts);
end

%% ---------------- Gradient and cost (devectorized, precomputed AB) ----------------
function [cost, grad_u] = stl_cost_devec_pre(u, A, B, g, z0, phi_str, C, dim_u, N, AB_cells)
    u_mat = reshape(u, dim_u, N);
    dim_z = length(z0);
    Z = zeros(dim_z, N);

    % Step-by-step Koopman propagation
    z = z0;
    for k = 1:N
        z = A*z + B*u_mat(:,k);
        Z(:,k) = z;
    end

    % Project to original states
    X = C*Z;

    % Wrap into struct for STL
    dim_x = size(C,1);
    x_struct = struct();
    max_vars = min(dim_x,10);
    for j = 1:max_vars
        x_struct.(sprintf('x%d', j)) = X(j,:);
    end

    % Compute STL robustness and gradient
    % [rob, grad_struct] = eval_stl_recursive(phi_str, x_struct);
    [rob, grad_struct] = robustness_gradient_fd(phi_str, x_struct);

    % Convert grad_struct to matrix
    grad_x = zeros(dim_x, N);
    for j = 1:max_vars
        grad_x(j,:) = grad_struct.(sprintf('x%d', j));
        % disp(grad_struct.(sprintf('x%d', j)));
        % keyboard;
    end

    % Gradient w.r.t lifted states
    grad_z = C' * grad_x;

    % Compute gradient w.r.t inputs using precomputed AB_cells
    grad_u_mat = zeros(dim_u, N);
    for j = 1:N
        for k = j:N
            grad_u_mat(:,j) = grad_u_mat(:,j) + (AB_cells{k-j+1})' * grad_z(:,k);
        end
    end

    grad_u = grad_u_mat(:);
    % disp('start');
    % disp(grad_u);
    % keyboard;
    cost = rob;
end


% Define custom OutputFcn that stops when fval < 0
function stop = stop_when_negative(x, optimValues, state)
    stop = false;  % default: continue
    if strcmp(state, 'iter')  % only check at each iteration
        if optimValues.fval < 0
            stop = true;
            fprintf('Stopping early: robustness dropped below 0 (%.4f).\n', optimValues.fval);
        end
    end
end
