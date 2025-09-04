% Time
t = 0:0.1:5;

% Signals
x_struct.x1 = sin(t);
x_struct.x2 = cos(t);

% STL Formula (using [t])
phi_str = 'always_[0,5](x1[t] + 2*x2[t] <= 1)';
phi = STL_Formula('phi', phi_str);


% Compute robustness and gradient
[rob, grad] = eval_stl_recursive(phi, x_struct);
disp(rob)
disp(grad)

% Plot robustness
figure;
plot(t, rob, 'LineWidth', 2);
xlabel('Time'); ylabel('Robustness');
title('STL Robustness');

% Plot gradient for x1
figure;
plot(t, grad.x1, 'r', 'LineWidth', 2); hold on;
plot(t, grad.x2, 'b', 'LineWidth', 2);
legend('∂Rob/∂x1', '∂Rob/∂x2');
xlabel('Time'); ylabel('Gradient');
title('Gradient of STL Robustness');
