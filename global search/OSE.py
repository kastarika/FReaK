import matlab.engine

# Start MATLAB engine
eng = matlab.engine.start_matlab()

def OSE(init_state, max_time, time_step, model, c, lb, ub, omega, CR, max_iter=50, level=1, select_dims=[0], input_dims=1):
  steps = max_time / time_step
  init_inputs = np.zeros((int(steps),int(input_dims)))
  init_outputs = model(init_state, init_inputs, time_step)
  traj_list = [(init_inputs, init_outputs)]
  
  for _ in range(max_iter):
    print(_, len(traj_list))
    target_features = []
    for j in range(level):
      feature = []
      for i in select_dims:
        t = np.random.randint(0, steps)
        dif = (ub[t][i] - lb[t][i])
        new_f = (1 + 2 * c) * dif * np.random.rand() + lb[t][i] - c * dif
        feature.append((new_f, t, i, dif))
      feature = np.array(feature)
      target_features.append(feature)
    target_features = np.array(target_features)

    #print(target_features)
    #print('++++++++++++++++++++++++++++++++++++')
    closest_traj = min(traj_list, key=lambda traj: dist(traj[1], target_features))
    #print(closest_traj)
    # closest_inputs = closest_traj[0]
    # new_inputs = closest_inputs.deepcopy()
    # for i, u in enumerate(closest_inputs):
    #   delta = omega * np.tan(np.pi * (np.uniform(-1,1) - 0.5))
    #   if(np.random.rand() < CR):
    #     new_inputs[i] = u + delta
    #   else:
    #     new_inputs[i] = u
    closest_inputs = closest_traj[0]
    #print(closest_inputs)
    #print('++++++++++++++++++++++++++++++++++++')
    new_inputs = copy.deepcopy(closest_inputs)
    delta = omega * np.tan(np.pi * (np.random.uniform(-1, 1, size=closest_inputs.shape) - 0.5))
    mask = np.random.rand(*closest_inputs.shape) < CR
    new_inputs[mask] = closest_inputs[mask] + delta[mask]
    #print(new_inputs)
    #print('++++++++++++++++++++++++++++++++++++')
    new_outputs = model(init_state, new_inputs, time_step)
    traj_list.append((new_inputs, new_outputs))
  return traj_list
