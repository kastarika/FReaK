function [u_opt, fval, exitflag] = optimize_koopman_gd_mem(A, B, g, x0, phi_str, u0, lb, ub, N)
    % Precompute A^p * B
    AB_cells = precompute_ApB(A, B, N);
    C = [eye(n_x), zeros(n_x, n_z-n_x)];

    % fmincon options
    opts = optimoptions('fmincon','Algorithm','sqp','Display','iter',...
                        'SpecifyObjectiveGradient',true,'MaxIterations',100);

    % Objective wrapper
    fun = @(u) stl_cost_koopman_mem(u, A, B, g, x0, phi_str, N, C, AB_cells);

    [u_opt, fval, exitflag] = fmincon(fun, u0, [], [], [], [], lb, ub, [], opts);
end

%% -------------------- Gradient and cost (memory-efficient) --------------------
function [cost, grad_u] = stl_cost_koopman_mem(u, A, B, g, x0, phi_str, N, C, AB_cells)
    dim_u = size(B,2);
    u_mat = reshape(u, dim_u, N);

    % Propagate Koopman trajectory
    z = g(x0);
    dim_z = length(z);
    Z = zeros(dim_z, N);
    for k = 1:N
        z = A*z + B*u_mat(:,k);
        Z(:,k) = z;
    end

    % Project to original state
    X = C*Z;

    % Map trajectory into STL struct
    max_vars = min(size(X,1),10);
    x_struct = struct();
    for j = 1:max_vars
        x_struct.(sprintf('x%d', j)) = X(j,:);
    end

    % STL robustness and gradient
    [rob, grad_struct] = eval_stl_recursive(phi_str, x_struct);

    % Convert grad_struct to matrix
    grad_x = zeros(size(X));
    for j = 1:max_vars
        grad_x(j,:) = grad_struct.(sprintf('x%d', j));
    end

    % Map STL gradient back to lifted space
    grad_z = C' * grad_x;

    % Compute grad_u without building full block matrix
    grad_u_mat = zeros(dim_u, N);
    for j = 1:N
        % vectorized sum over k >= j
        ks = j:N;
        AB_stack = cat(3, AB_cells{1:(N-j+1)});      % dim_z x dim_u x num_future
        AB_stack = AB_stack(:,:,ks-j+1);             % select relevant
        grad_z_stack = grad_z(:,ks);                 % dim_z x num_future
        % Multiply each A^p*B' by corresponding grad_z
        for idx = 1:length(ks)
            grad_u_mat(:,j) = grad_u_mat(:,j) + AB_stack(:,:,idx)' * grad_z_stack(:,idx);
        end
    end

    grad_u = grad_u_mat(:);
    cost = rob;
end
