% generator for trajectories produced by OSE
% takes in the name of the OSE producer for the model
% the OSE producer takes in 
function next = next_traj_OSE(model)
    persistent traj_list idx
    next = []; % Initialize next to an empty array if no trajectory is available
    if isempty(traj_list) | idx > length(traj_list)
        traj_list = feval(model.OSE); % Reset the trajectory list
        idx = 1; % Reset index
    end
    input = traj_list{idx}{1};
    output = traj_list{idx}{2};
    % first_row = output(1,:).';
    first_row = model.R0.inf;
    rest = input(2:end,:);
    rest_flat = rest(:);
    next = [first_row; rest_flat]; % Get the next trajectory
    idx = idx + 1; % Increment index for next call
end
