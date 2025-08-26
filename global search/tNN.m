function ose = tNN()
    ose = OSE();
    % Simulation settings
    ose.max_time = 40;
    ose.time_step = 40/3;
    ose.input_dims = 1;
    ose.output_dims = 2;

    % Bounds
    ose.lb = zeros(ose.max_time + 1, ose.output_dims);
    ose.ub = ones(ose.max_time + 1, ose.output_dims);
    ose.lb(:) = -30;
    ose.ub(:) = 30;
    
    % OSE parameters
    ose.c = 0.2;
    ose.omega = 0.2;
    ose.CR = 0.6;
    ose.max_iter = 50;
    ose.select_dims = [1, 2];
    ose.simu_name = 'narmamaglev_v1';  
    ose.ro = [4/7, 2/7, 1/7];
    
    % ose.stl_req = 'alw_[0,19]((alw_[0,5](b_1[t] <= 20) or (ev_[0,5](b_2[t] >= 40))))';
    %ose.stl_req = 'alw_[0,65](ev_[0,30](alw_[0,5](b_5[t]-b_4[t]>=8)))';
    ose.run_OSE();
    % ose.run_OSE();
    % ose.run_OSE();
end



