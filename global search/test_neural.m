% Initialize variables
evalin("base","init_neural;")

% Define input signal
t__ = [0; 10; 20; 30; 40];
u__ = [1;  2;  2; 2.5; 1.5];
u = [t__, u__];

% Define time limit
T = 40;

% Run model
[tout, yout] = test_neural_run(u, T);