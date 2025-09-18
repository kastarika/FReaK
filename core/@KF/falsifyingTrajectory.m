function [x0,u] = falsifyingTrajectory(obj,soln)
    u=soln.u;
    x0=soln.x0;
    all_steps = obj.T/obj.ak.dt;
    % disp(all_steps)
    % keyboard
    %check that correct number of inputs is returned, sometimes we have a
    %final dummy input so that number of inputs is equal to state variables
    %for MILP stl encoding, we remove it.
    disp('akjsdhfkjasdhfjkahsdkjfhakjdshfjk')
    disp(size(u))
    if size(u,2) == all_steps
    elseif size(u,2)==all_steps+1 %inputs returned include last time step
        u=u(:,1:end-1);
    else
        error('incorrect number of inputs returned from solver, investigate error')
    end
    tp_=linspace(0,obj.T-obj.ak.dt,all_steps); %time points without last time step
    tp = linspace(0,obj.T,all_steps+1);
    u = interp1(tp_',u',tp',obj.inputInterpolation,"extrap"); %interpolate and extrapolate input points
    u =  max(obj.U.inf',min(obj.U.sup',u)); %ensure that extrapolation is within input bounds
    u = [tp',u];
end