
% algorithm settings
numFeat = 50;           % number of observables
l = 1;                  % lengthscale parameter
dt = 1;                 % time step for the Koopman model
Ntraj = 10;             % number of trajectories

% orignal system dynamics
f = @(x,u) [-(x(2)+x(3)) + u(1); ...
            x(1) + 0.2*x(2) + u(2); ...
            0.2 + x(3)*(x(1) - 5.7)];

sysOrig = nonlinearSys(f);

params.R0 = interval([-0.05;-8.45;-0.05],[0.05;-8.35;0.05]);
params.U = 0.1*interval([-1;-1],[1;1]);
params.tFinal = 6;

% specification ("stay in the region x(1) <= -6 for at least 2 seconds")
x = stl('x',sysOrig.nrOfDims);
phi = finally(globally(x(1) <= -6,interval(0,1)),interval(0,6));

% generate data
x = cell(Ntraj,1); u = cell(Ntraj,1); t = cell(Ntraj,1);

for i = 1:Ntraj

    x{i} = randPoint(params.R0)';

    for j = 1:floor(params.tFinal/dt)
    
        simOpts.x0 = x{i}(end,:);
        simOpts.u = randPoint(params.U);
        simOpts.tFinal = dt;

        [~,xTmp] = simulate(sysOrig,simOpts);

        x{i} = [x{i};xTmp(end,:)];
        u{i} = [u{i};simOpts.u'];
    end

    t{i} = (0:dt:params.tFinal)';
end

% identify a suitable Koopman model
[A,B,c,g] = identifyKoopmanModel(x,t,u,numFeat,l);

% falsify the Koopman model via gradient-based optimization
fals = falsifyKoopmanGradient(A,B,c,g,params.R0,params.U,dt,phi);

% visualize the result
figure; hold on; box on;
plot(fals.x(1,:),fals.x(2,:));
plot(polytope([1,0,0],-6),[1,2],'FaceColor','r','FaceAlpha',0.5,'EdgeColor','none');
plot(params.R0,[1,2],'FaceColor','none','EdgeColor','k');

sysKoop = linearSysDT(A,B,c,dt);

for i = 1:length(x)
    simOpts.x0 = g(x{i}(1,:)');
    simOpts.tFinal = params.tFinal;
    simOpts.u = u{i}';
    [~,xKoop] = simulate(sysKoop,simOpts);
    plot(xKoop(:,1),xKoop(:,2),'k');
end

plot(fals.x(1,:),fals.x(2,:),'g','LineWidth',1);