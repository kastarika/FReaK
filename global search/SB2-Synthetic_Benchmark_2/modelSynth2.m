function model = modelSynth2()
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
    
    model = KF('phi2_m1_vr01_k2_2');
    model.R0 = interval(10.4352, 10.4352); 
    model.U = interval(0,1); 

    model.T=30; 
    model.dt = 0.01;
    model.ak.dt=1; %2.5
    model.cp=[5];
    
    model.nResets = 4;

    x = stl('x',1);
    eq = globally(x(1) > 90 | finally(x(1) < 50, interval(0,6)), interval(0,18));
%     eq = globally(x(1) < 120,interval(0,20));
%         eq = finally(x(1) > 120,interval(0,20));
%     eq = globally(x(2) < 4750,interval(0,10)) | globally(x(1)<50,interval(0,10));
%     eq = until(x(2) < 4500, x(1)>100,interval(0,30));
%     eq = next(x(2)<3870, 10);
    model.spec = specification(eq,'logic');
%     model.spec = specification(halfspace([0 -1 0],-4750),'unsafeSet');
%     model.spec = specification(halfspace([-1 0 0],-120),'unsafeSet');

end