function fals = falsifyKoopmanGradient(A,B,c,g,X0,U,dt,phi)
% falsify a Koopman model g(x(k+1)) = A*g(x(k)) + B*u(k) + c by minimizing 
% the robustness of the STL formula phi via gradient-based optimization

    % initialization
    n = X0.dim; m = U.dim;
    tFinal = maximumTime(phi);
    N = floor(tFinal/dt);
    t = 0:dt:N*dt;

    % precompute gradient of the observable function
    x = sym("x",[n,1]);
    gSym = g(x);
    dg = matlabFunction(jacobian(gSym,x),'Vars',{x});

    % construct propagation matrices for all time steps, such that 
    % x(ti) = P{i}.A * g(x0) + P{i}.B * u + P{i}.c 
    P = cell(N,1); len = m*N; 
    A_ = eye(size(A)); B_ = []; c_ = zeros(size(A,1),1);

    for i = 1:N

        % update matrices
        A_ = A*A_;
        c_ = A*c_ + c;

        if ~isempty(B_)
            B_ = [A*B_,B];
        else
            B_ = B;
        end

        % store propagation matrices
        P{i}.A = A_;
        P{i}.B = [B_(1:n,:),zeros(n,len-size(B_,2))];
        P{i}.c = c_(1:n);
    end

    % constuct initial state
    x0 = center(X0); u = center(U);
    z0 = [x0;repmat(u,[N,1])];

    % construct constraints x0 \in X0 and \forall t: u(t) \in U
    X0 = polytope(X0); U = polytope(U);

    tmp = [{X0.A},repmat({U.A},[1,N])];
    A = blkdiag(tmp{:});
    b = [X0.b;repmat(U.b,[N,1])];

    % minimize robustness via gradient-based optimization
    options = optimoptions('fmincon','Display','off', ...
                            'SpecifyObjectiveGradient',true);

    z = fmincon(@(z) aux_gradientKoopman(z,P,g,dg,phi,n,t), ...
                                            z0,A,b,[],[],[],[],[],options);

    % simulte Koopman model to obtain the final trajectory
    x0 = z(1:n); u = z(n+1:end);

    x = zeros(n,length(P)+1);
    xInit = g(x0); x(:,1) = xInit(1:n);

    for i = 1:length(P)
        x(:,i+1) = P{i}.A(1:n,:)*xInit + P{i}.B*u + P{i}.c; 
    end

    fals.x0 = x0; fals.u = reshape(u,[m,N]); fals.t = t; fals.x = x;
end

% Auxiliary functions -----------------------------------------------------

function [c,g] = aux_gradientKoopman(z,P,g,dg,phi,n,t)

    % split optimization variables into initial states and inputs
    x0 = z(1:n);
    u = z(n+1:end);

    dg = dg(x0);

    % compute the current trajectory of the Koopman model
    x = zeros(n,length(P)+1);
    xInit = g(x0); x(:,1) = xInit(1:n);

    for i = 1:length(P)
        x(:,i+1) = P{i}.A(1:n,:)*xInit + P{i}.B*u + P{i}.c; 
    end

    % compute the gradient of the STL specification
    [c,gr] = gradientTemporalLogic(phi,x,t);

    % combine gradient with the dynamics of the Koopman model
    g = [gr(1,:)*dg(1:n,:),zeros(1,length(u))];

    for i = 1:length(P)
        tmp = P{i}.A*dg;
        g = g + gr(i+1,:)*[tmp(1:n,:),P{i}.B];
    end
end