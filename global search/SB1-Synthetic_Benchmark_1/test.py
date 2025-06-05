import matlab.engine

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define variables
t = matlab.double([[0], [6], [12], [18], [24]])
u_vals = matlab.double([
    [0.0, 0.75],
    [0.25, 0.25],
    [0.5, 0.75],
    [0.75, 0.25],
    [1.0, 0.75]
])

# Concatenate t and u horizontally
# MATLAB-style: u = [t, u]
u_combined = eng.horzcat(t, u_vals)
#u_combined = matlab.double([t,u_vals])
# Empty array for p
p = matlab.double([])

# Simulation time
T = 24

# Call the MATLAB function
tout, yout = eng.run_synth_benchmark1(p, u_combined, T, nargout=2)

# Convert output to Python list if needed
tout_py = list(tout)
yout_py = [list(row) for row in yout]

# Optional: print results
print("Time:", tout_py)
print("Output:", yout_py)
