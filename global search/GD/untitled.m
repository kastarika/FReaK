% Time vector
t = 0:0.1:5;

% Create signal structure
x_struct.x1 = sin(t);
x_struct.x2 = cos(t);

% Example predicate: x1[t] + 2*x2[t] <= 0.5
expr = 'x1[t] + 2*x2[t] <= 0.5';

% Evaluate robustness and gradient
[rob, grad] = eval_predicate(expr, x_struct);

% Plot robustness
figure;
plot(t, rob, 'LineWidth', 2);
title('Predicate Robustness over Time');
xlabel('Time'); ylabel('Robustness');
