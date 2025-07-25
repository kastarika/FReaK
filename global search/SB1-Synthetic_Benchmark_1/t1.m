function ose = t1()
    ose = OSE();
    % Simulation settings
    ose.max_time = 30;
    ose.time_step = 6;
    ose.input_dims = 2;
    ose.output_dims = 1;
    
    % Bounds
    ose.lb = zeros(ose.max_time + 1, ose.output_dims);
    ose.ub = ones(ose.max_time + 1, ose.output_dims);
    ose.lb(:) = -20;
    ose.ub(:) = 25;
    
    % OSE parameters
    ose.c = 0.2;
    ose.omega = 0.2;
    ose.CR = 0.6;
    ose.max_iter = 100;
    ose.select_dims = [1];
    ose.simu_name = 'phi1_m2_vr001_k5_2';  
    ose.ro = [2/7, 2/7, 2/7, 1/7];
    
    % ose.stl_req = '(ev_[6,12] (b_1[t]>10)) => (alw_[18,24] (b_1[t]>-10))';
    ose.stl_req = '(alw_[0,24](b_1[t]<20))';

    ose.run_OSE(true);
    
end



