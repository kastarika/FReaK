function [A,B,c,g] = identifyKoopmanModel(x,t,u,numFeat,l)
% identify a Koopman model that fits the given data

    n = size(x{1},2);

    % generate observable function
    g = aux_randomFourierFeatureObservables(numFeat,n,l);

    % transform data by the observable function
    x_ = cell(size(x));

    for i = 1:length(x)
        for j = 1:size(x{i},1)
            x_{i} = [x_{i};g(x{i}(j,:)')'];
        end
    end

    x = x_;

    % identify a suitable Koopman model    
    sys = aux_identifyDMD(x,t,u);

    A = sys.A; B = sys.B; c = sys.c;
end


% Auxiliary functions -----------------------------------------------------

function g = aux_randomFourierFeatureObservables(numFeat,dim,l)
% generate Random Fourier Feature observables cos(w'*x + u)

    % generate random scales and offsets
    w = normrnd(0,l^2,numFeat,dim);
    u = 2*pi*rand(numFeat,1);
    
    % generate fourier transform observables
    g = @(x) [x; sqrt(2)*cos(w*x + u)];
end

function sys = aux_identifyDMD(x,t,u)
% identify linear discrete-time model via Dynamic Mode Decomposition (DMD)

    % bring data to the correct format
    data = cell(size(x));

    for i = 1:length(data)
        data{i}.x = x{i};
        data{i}.t = t{i};
        data{i}.u = u{i};
    end

    % split the data into single data points
    points = aux_getDataPoints(data);

    % apply dynamic mode decomposition (DMD) for all ranks of the SVD
    Alist = aux_dynamicModeDecomposition(points.x',points.xNext',points.u');

    % select the matrix that best fits the data
    errBest = inf;
    dt = data{1}.t(2) - data{1}.t(1);

    for i = 1:length(Alist)
        [err,sysTemp] = aux_computeError(Alist{i},data,dt);
        if err < errBest
            sys = sysTemp;
            errBest = err;
        end
    end 
end

function A = aux_dynamicModeDecomposition(X1,X2,U)
% compute the matrix X2 = A*X1 that best fits the data using the approach
% in Equation (2.7) in [1]

    X1 = [X1; ones(1,size(X1,2))];

    if ~isempty(U)
        X1 = [X1; U];
    end

    % singular value decomposition
    [V,S,W] = svd(X1,'econ');
    
    % construct matrices with different rank
    rankMax = sum(diag(S) > 0);
    A = cell(rankMax,1);

    for rank = 1:rankMax

        % reduce rank by removing the smallest singular values
        if ~isempty(rank) && rank < size(S,1)
            V_ = V(:,1:rank); S_ = S(1:rank,1:rank); W_ = W(:,1:rank);
        else
            V_ = V; S_ = S; W_ = W;
        end
    
        % compute resulting system matrix A
        A{rank} = X2*W_*diag(1./diag(S_))*V_';
    end
end

function [err,sys] = aux_computeError(Aall,data,dt)
% compute the error between the system approximation and the real data

    err = 0;

    % construct linear system object
    n = size(Aall,1);
    A = Aall(:,1:n); c = Aall(:,n+1); B = Aall(:,n+2:end); 

    sys = linearSysDT(A,B,c,dt);

    % loop over all trajectories
    for i = 1:length(data)

        % simulation options
        simOpts = [];
        simOpts.x0 = data{i}.x(1,:)';
        simOpts.tStart = data{i}.t(1);
        simOpts.tFinal = data{i}.t(end);
        if isfield(data{i},'u')
            simOpts.u = data{i}.u';
        end

        % simulate the system
        [~,x] = simulate(sys,simOpts);

        % compute the error for the current trajectory
        err = err + mean(sum((x-data{i}.x).^2,2));
    end
end

function points = aux_getDataPoints(traj)
% transform the data into a list of data points

    points.x = [];
    points.u = [];
    points.xNext = [];
    points.dt = [];

    for i = 1:length(traj)

        m = size(traj{i}.x,1)-1;

        points.x = [points.x; traj{i}.x(1:end-1,:)];
        points.xNext = [points.xNext; traj{i}.x(2:end,:)];
        points.dt = [points.dt; diff(traj{i}.t)];
        
        if isfield(traj{i},'u')
            points.u = [points.u; traj{i}.u(1:m,:)];
        end
    end
end