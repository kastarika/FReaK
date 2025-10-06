function koopModel = getautokoopman(trajes, obj)
    trainset = struct();
    trainset.t = {};
    trainset.X = {};
    trainset.XU = {};
    trainset.Rob = {};
    for i = 1:length(trajes)
        traj = trajes{i};
        x = traj.x;
        u = traj.u;
        tsim = traj.t;
        tak = (0:obj.ak.dt:obj.T)';
        % tcp = (0:obj.dt:obj.T)';
        xak = interp1(tsim,x,tak,obj.trajInterpolation);
        trainset.t{end+1} = tak;
        trainset.X{end+1} = xak';
        trainset.XU{end+1} = u(:,2:end)';
        trainset.Rob{end+1} = 1;
       
    end
    % keyboard
    % [koopModel,koopTime] = learnKoopModel(obj, trainset(1));
    [koopModel,koopTime] = learnKoopModel(obj, trainset);
end