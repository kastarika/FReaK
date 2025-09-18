function koopman = getautokoopman(trajes, obj)

    for i = 1:length(trajes)
        traj = trajes{i};
        x = traj.x;
        t = traj.t;
        tak = (0:obj.ak.dt:obj.T)';
        tcp = (0:obj.dt:obj.T)';
        xak = interp1(tsim,x,tak,obj.trajInterpolation);
        
    end

end