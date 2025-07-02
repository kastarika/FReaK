function ose = t2()
    ose = OSE();
    % Simulation settings
    ose.max_time = 24;
    ose.time_step = 6;
    ose.input_dims = 1;
    ose.output_dims = 1;
    
    % Bounds
    ose.lb = zeros(ose.max_time + 1, ose.output_dims);
    ose.ub = ones(ose.max_time + 1, ose.output_dims);
    % ose.lb(:) = 30;
    % ose.ub(:) = 100;

    
    % OSE parameters
    ose.c = 0.2;
    ose.omega = 0.03;
    ose.CR = 0.8;
    ose.max_iter = 200;
    ose.select_dims = [1];
    ose.simu_name = 'phi2_m1_vr01_k2_2';  
    ose.ro = [4/7, 2/7, 1/7];
    
    % ose.stl_req = '(ev_[6,12] (b_1[t]>10)) => (alw_[18,24] (b_1[t]>-10))';
    ose.stl_req = 'alw_[0,18] (b_1[t] > 90 or (ev_[0,6] (b_1[t] < 50)))';

    ose.run_OSE_input();
    
end



