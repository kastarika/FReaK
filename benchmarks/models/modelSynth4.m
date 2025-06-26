function model = modelSynth4()
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
    
    model = KF('phi4_m2_vr001_k5_3');
    model.R0 = interval([2.597;14.902],[2.597;14.902]); 
    model.U = interval([0;0],[1;1]); 

    model.T=24; 
    model.dt = 0.01;
    model.ak.dt=3; %2.5
    model.cp=[4 4];

    x = stl('x',2);
    eq = globally((globally(x(1)<=20, interval(0, 5))) | (finally(x(2)>=40, interval(0, 5))), interval(0, 19));
%     eq = globally(x(1) < 120,interval(0,20));
%         eq = finally(x(1) > 120,interval(0,20));
%     eq = globally(x(2) < 4750,interval(0,10)) | globally(x(1)<50,interval(0,10));
%     eq = until(x(2) < 4500, x(1)>100,interval(0,30));
%     eq = next(x(2)<3870, 10);
    model.spec = specification(eq,'logic');
%     model.spec = specification(halfspace([0 -1 0],-4750),'unsafeSet');
%     model.spec = specification(halfspace([-1 0 0],-120),'unsafeSet');

end