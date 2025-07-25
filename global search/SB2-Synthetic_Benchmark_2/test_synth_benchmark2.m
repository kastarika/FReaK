% No variables to initialize

% Define input signals
t = [0; 6; 12; 18; 24];
% u = [0.0; 0.25; 0.5; 0.75; 1.0];
u = [-0.59353;
0.20947;
0.19124;
0.065885;
-0.1762;
];

% Define total simulation time
p = [];
u = [t, u];
T = 24;

% Run model
[tout, yout] = run_synth_benchmark2(p, u, T);