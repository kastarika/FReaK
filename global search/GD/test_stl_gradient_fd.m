% ---------------------------
% Test robustness_gradient_fd
% ---------------------------

% Define trajectories for 3 states
T = 20;
t = 0:T-1;

X.x1 = sin(0.2*t) + 0.1*randn(1,T);   % noisy sine
X.x2 = cos(0.2*t) + 0.2;              % shifted cosine
X.x3 = linspace(0, 5, T);             % increasing ramp

% Define STL requirement:
%   φ = "Always between t=0 and t=15, if x1 > 0 then eventually within 5 steps x2 < 0.5,
%        and x3 stays below 4 for the entire horizon"
%
% In STL (Breach syntax):
phi_str = 'alw ( (x1 > 0) => ev_[0,5] (x2 < 0.5) ) and alw (x3 < 4)';

% Compute robustness and gradient
[rob, grad] = robustness_gradient_fd(phi_str, X);

% Display results
disp('Robustness of the trace:');
disp(rob);

disp('Gradient wrt x1:');
disp(grad.x1);

disp('Gradient wrt x2:');
disp(grad.x2);

disp('Gradient wrt x3:');
disp(grad.x3);

% Plot original signals for intuition
figure;
subplot(3,1,1);
plot(t, X.x1, 'b-o'); ylabel('x1');
title('Test Trajectories');
subplot(3,1,2);
plot(t, X.x2, 'r-o'); ylabel('x2');
subplot(3,1,3);
plot(t, X.x3, 'g-o'); ylabel('x3'); xlabel('time');
