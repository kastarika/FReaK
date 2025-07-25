function model = modelSynth1()
% model_AutoTransmission - model parameters for the Autotransmission benchmark
%
% Syntax:
%       model = model_AutoTransmission()
%
% Description:
%       Model parameters for the vanderpol benchmark.
%
% Output Arguments:
%
%       -model:             a koopman falsification model      
%
%------------------------------------------------------------------
    
    model = KF('phi1_m2_vr001_k5_2');
    model.R0 = interval(1.5806,1.5806); 
    model.U = interval([0;0],[1;1]); 

    model.T=30; 
    model.dt = 0.01;
    model.ak.dt=1; %2.5
    model.cp=[5 5];
    
    model.nResets = 5;
    % model.resetStrat = 0;
    % model.trainStrat = 2;
    model.rmRand = false;
    % model.reach.on = false;

    x = stl('x',1);
    % eq = implies(finally(x(1) > 10, interval(6, 12)), globally(x(1) > -10, interval(18, 24)));
    eq = globally(x(1) < 20, interval(0,24));
    % eq = (F[6,12](x1 > 10)) => (G[18,24](x1 > -10));
    % eq = '(ev_[6,12] (b_1[t]>10)) => (alw_[18,24] (b_1[t]>-10))';
    % eq = implies(globally(x(1) >=250 & x(1) <=260,interval(1,1.5)),globally(x(1)<230|x(1)>240,interval(3,4)));

%     eq = globally(x(1) < 120,interval(0,20));
%         eq = finally(x(1) > 120,interval(0,20));
%     eq = globally(x(2) < 4750,interval(0,10)) | globally(x(1)<50,interval(0,10));
%     eq = until(x(2) < 4500, x(1)>100,interval(0,30));
%     eq = next(x(2)<3870, 10);
    model.spec = specification(eq,'logic');
%     model.spec = specification(halfspace([0 -1 0],-4750),'unsafeSet');
%     model.spec = specification(halfspace([-1 0 0],-120),'unsafeSet');

end