% No variables to initialize

% Define input signals
t = [0; 6; 12; 18; 24; 30];
u = [0.0 0.0638; 1.0000 0.2789; 0.3302 1.0000; 0.7983 0.7504; 0.7357 0.1312; 0.2383 0.3079];

% Define total simulation time
p = [];
u = [t, u];
T = 30;

% Run model
[tout, yout] = run_synth_benchmark1(p, u, T);