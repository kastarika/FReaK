%% Define a simple trajectory struct
T = 5; % number of time steps
x_struct = struct();
for i = 1:3
    x_struct.(['x' num2str(i)]) = linspace(0,1,T);  % 3 states
end

%% Define a simple STL formula in Breach format
% For example: always (x1 - x2 > 0) && eventually (x3 < 0.5)
phi = 'always(x1 - x2 > 0) and eventually(x3 < 0.5)';

%% Compute robustness and gradient
[rob, grad] = stl_robustness_fd(phi, x_struct);

%% Display
disp('Robustness:')
disp(rob)

disp('Gradient:')
fields = fieldnames(grad);
for i = 1:length(fields)
    fprintf('%s: %s\n', fields{i}, mat2str(grad.(fields{i})))
end


eps = 1e-6;
x_test = x_struct;
fd_grad = zeros(3,T);
fields = fieldnames(x_struct);
for f = 1:3
    for t = 1:T
        x_plus = x_struct;
        x_plus.(fields{f})(t) = x_plus.(fields{f})(t) + eps;
        rob_plus = eval_stl_recursive(phi, x_plus);
        disp('adsfsdf')
        disp(rob_plus)
        disp(rob)
        fd_grad(f,t) = (rob_plus - rob) / eps;
    end
end

disp('Finite difference gradient:')
disp(fd_grad)

disp('Analytical gradient from eval_stl_recursive:')
for f = 1:3
    disp(grad.(fields{f}))
end
