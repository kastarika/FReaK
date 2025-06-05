import matlab.engine

# Start MATLAB
eng = matlab.engine.start_matlab()

# Define input time vector
t = matlab.double([[0], [6], [12], [18], [24]])

# Define input values (single column)
u_vals = matlab.double([[0.0], [0.25], [0.5], [0.75], [1.0]])

# Combine into u = [t, u]
u_combined = eng.horzcat(t, u_vals)

# Empty parameter list
p = matlab.double([])

# Simulation time
T = 24

# Call the MATLAB function that wraps the Simulink model
# This assumes run_synth_benchmark2 sets StopTime properly
#tout, yout = eng.run_synth_benchmark2(p, u_combined, T, nargout=2)
eng.test_synth_benchmark2()
# (Optional) Convert MATLAB outputs to Python-friendly formats
#tout_py = list(tout)
#yout_py = [list(row) for row in yout]

# Print or use results
#print("Time:", tout_py)
#print("Output:", yout_py)
