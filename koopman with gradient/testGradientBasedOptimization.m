% algorithm settings
numFeat = 50;          % number of observables
l = 1;                 % lengthscale parameter
maxFunEval = 400;      % max. number function evaluation for optimization
maxIter = 500;         % max. number of iterations for optimization
output_dim = 5;        % number of outputs of the system

% generate Random Fourier Feature observables
% g = randomFourierFeatureObservables(numFeat,2,l);
g = randomFourierFeatureObservables(numFeat,output_dim,l);
% 5 = number of outputs the system has

tic;
% generate data
% f = @(x,u) [x(2); ...
%             (1-x(1)^2)*x(2)-x(1)];
% 
% sysOrig = nonlinearSys(f);
% 
% params.tFinal = 10;
% params.R0 = zonotope(interval([-3;-3],[3;3]));
% 
% options.points = 6;
% 
% simRes = simulateRandom(sysOrig,params,options);

L = 20;

T = 100;
time_step = 5;
steps = T / time_step + 1;
timestamps = (0:steps - 1)' * time_step;
simu_name = 'cars';
input_dims = 2;
p = [];

params.tFinal = T;

trajes = cell(L, 1);
disp(size(trajes));

for i = 1:L
    u = rand(steps, input_dims);
    trajes{i}.u = u;
    u = [timestamps, u];
    [tout, yout] = runSimu(simu_name, T, p, u);
    trajes{i}.t = tout;
    trajes{i}.x = yout;
    % disp(size(tout))
    % disp(size(yout))
end
keyboard;

% transform data by the observable function
x = cell(L,1);
t = cell(L,1);

for i = 1:L
    % t{i} = simRes(1).t{i};
    % x_ = simRes(1).x{i};
    t{i} = trajes{i}.t;
    x_ = trajes{i}.x;
    disp(size(x_));
    for j = 1:size(x_,1)
        x{i} = [x{i};g(x_(j,:)')'];
    end
end

% disp()

% identify a Koopman model (optimization)
% p = size(simRes(1).x{1},2);
p = size(trajes{1}.x,2);


data=cell(size(x));
for i=1:L
    data{i}.t = t{i};
    data{i}.x = x{i};
end


sysOpt = aux_identifyOpt(x(1:L),t(1:L),p,maxFunEval,maxIter);

% identify a Koopman model (DMD)
% optOpts.alg = 'dmd';
% sysDMD = aux_identifyDMD(data);



% simulate the Koopman model
simResOpt = [];
% simResDMD = [];

for i = 1:L

    % simOpts.x0 = g(simRes(1).x{i}(1,:)');
    simOpts.x0 = g(trajes{i}.x(1,:)');
    simOpts.tFinal = ceil(params.tFinal/sysOpt.dt)*sysOpt.dt;

    [tOpt,xOpt] = simulate(sysOpt,simOpts);
    % [tDMD,xDMD] = simulate(sysDMD,simOpts);

    simResOpt = [simResOpt;simResult({xOpt},{tOpt})];
    % simResDMD = [simResDMD;simResult({xDMD},{tDMD})];
end

disp(simResOpt)
elapsed_time = toc;
disp(elapsed_time);
keyboard;

% visualization
figure; hold on; box on;
h1 = plot(simRes);
xx = simResOpt.x;
xx = xx{1};
xx = xx(:,1:2);
h2 = plot(xx(:,1), xx(:,2));
% h2 = plot(simResOpt);
% h3 = plot(simResDMD);

% legend([h1,h2,h3],'ground truth','optimization','dmd');

legend([h1,h2],'ground truth','optimization');



% Auxiliary functions -----------------------------------------------------

function g = randomFourierFeatureObservables(numFeat,dim,l)
% generate Random Fourier Feature observables cos(w'*x + u)

    % generate random scales and offsets
    w = normrnd(0,l^2,numFeat,dim);
    u = 2*pi*rand(numFeat,1);
    
    % generate fourier transform observables
    g = @(x) [x; sqrt(2)*cos(w*x + u)];
end

function sys = aux_identifyOpt(x,t,p,maxFunEval,maxIter)
% identifies a linear discrete-time system from trajectory data using 
% gradient-based optimization

    % bring data to the correct format
    data = cell(size(x));

    for i = 1:length(x)
        data{i}.t = t{i};
        data{i}.x = x{i};
    end

    % convert data to uniform time step size
    dt = aux_averageTimeStepSize(data);
    data = aux_uniformTimeStepSize(data,dt);

    % initialization
    n = size(data{1}.x,2);
    if isfield(data{1},'u')
        m = size(data{1}.u,2);
    else
        m = 0;
    end

    % determine initial guess via Dynamic Mode Decomposition (DMD)
    sys = aux_identifyDMD(data);

    A0 = [sys.A,sys.c];
    A0 = reshape(A0,[numel(A0),1]);

    % optimize using fmincon
    w = warning(); warning('off');

    options = optimoptions('fmincon','SpecifyObjectiveGradient',true, ...
                            'Display','iter','MaxIter',maxIter, ...
                            'MaxFunctionEvaluations',maxFunEval, ...
                            'PlotFcn','optimplotfval');

    Aall = fmincon(@(A) aux_costFunGrad(A,data,p),A0,[],[],[],[], ...
                                                        [],[],[],options);

    warning(w);

    % construct linear system object
    Aall = reshape(Aall,[n,n+m+1]);
    A = Aall(:,1:n); c = Aall(:,n+1); B = Aall(:,n+2:end); 

    sys = linearSysDT(A,B,c,dt);
end

function [cost,grad] = aux_costFunGrad(x,data,p)
% compute the value of the cost function together with the gradient

    % extract system matrices
    n = size(data{1}.x,2);
    m = length(x)/n - n;
    ns = n^2;
    Aall = reshape(x,[n,n+m]);
    A = Aall(:,1:n); B = Aall(:,n+1:end);

    % initialization
    cost = 0;
    grad = zeros(1,n*(n+m));

    % loop over all trajectories
    for i = 1:length(data)

        x = data{i}.x(1,:)';
        dx = zeros(n,n*(n+m));

        % loop over all time steps
        for j = 2:length(data{i}.t)
            
            xi = data{i}.x(j,:)';
            if isfield(data{i},'u')
                ui = [1;data{i}.u(j-1,:)'];
            else
                ui = 1;
            end

            % compute error
            x_prev = x;
            x = A*x + B*ui;

            cost = cost + sum((x(1:p) - xi(1:p)).^2);

            % compute gradient dx = U*dB + X*dA + A*dx (part A*dx)
            dx = A*dx;

            % compute gradient dx = U*dB + X*dA + A*dx (part X*dA)
            for k = 1:n
                for l = 1:n
                    dx(l,(k-1)*n+l) = dx(l,(k-1)*n+l) + x_prev(k);
                end
            end

            % compute gradient dx = U*dB + X*dA + A*dx (part U*dB)
            for k = 1:m
                for l = 1:n
                    dx(l,ns + (k-1)*n+l) = dx(l,ns + (k-1)*n+l) + ui(k);
                end
            end

            grad = grad + 2*(x(1:p) - xi(1:p))' * dx(1:p,:);
        end
    end
end

function dt = aux_averageTimeStepSize(data)
% compute the average time step size from the trajectory data
    
    % loop over all trajectories
    dt = 0; N = 0;

    for i = 1:length(data)

        % remove duplicate times
        [~,ind] = unique(data{i}.t);

        % add time steps for current trajectory
        dt = dt + sum(diff(data{i}.t(ind)));
        N = N + length(ind) - 1;
    end

    % compute average time step size
    dt = dt/N;
end

function data = aux_uniformTimeStepSize(data,dt)
% convert the trajectory data to a uniform time step size

    for i = 1:length(data)
        
        % check if the data already has the correct time step size
        d = diff(data{i}.t);

        if all(abs(d - dt) < eps)
            continue;
        end

        % interpolate to convert data to correct time step size
        [~,ind] = unique(data{i}.t);
        t = data{i}.t(1):dt:data{i}.t(end);

        data{i}.x = interp1(data{i}.t(ind),data{i}.x(ind,:),t, ...
                                                    'linear','extrap');

        if size(data{i}.x,1) ~= length(t)
            data{i}.x = data{i}.x';
        end

        if isfield(data{i},'u')
            data{i}.u = interp1(data{i}.t(ind),data{i}.u(ind,:),t, ...
                                                    'linear','extrap');
            if size(data{i}.u,1) ~= length(t)
                data{i}.u = data{i}.u';
            end
        end

        data{i}.t = t';
    end
end

function sys = aux_identifyDMD(data)
% identifies a linear discrete-time system from trajectory data using 
% Dynamic Mode Decomposition (DMD)

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
