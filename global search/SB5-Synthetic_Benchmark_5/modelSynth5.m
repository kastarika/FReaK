function model = modelSynth5()
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
    
    model = KF('phi5_m1_vr01_k5_2');
    model.R0 = interval([-1.3244;0.5447],[-1.3244;0.5447]); 
    model.U = interval(0,1);
    disp('akjdfhkjadhkfsjkakjaf');
    disp(length(model.U));
    model.T=24; 
    model.dt = 0.01;
    model.ak.dt=3; %2.5
    model.cp=[4];
    model.OSE='test_OSE5';
    x = stl('x',2);
    eq = globally(finally(~((globally(x(1)>=9, interval(0,1)) | (globally(x(2)>=9, interval(1,5)) ))) ,interval(0,2)) , interval(0, 17));
%     eq = globally(x(1) < 120,interval(0,20));
%         eq = finally(x(1) > 120,interval(0,20));
%     eq = globally(x(2) < 4750,interval(0,10)) | globally(x(1)<50,interval(0,10));
%     eq = until(x(2) < 4500, x(1)>100,interval(0,30));
%     eq = next(x(2)<3870, 10);
    model.spec = specification(eq,'logic');
%     model.spec = specification(halfspace([0 -1 0],-4750),'unsafeSet');
%     model.spec = specification(halfspace([-1 0 0],-120),'unsafeSet');

end