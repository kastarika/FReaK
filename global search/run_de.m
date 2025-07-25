max_time = 24;
time_step = 6;
% simu_name = 'phi1_m2_vr001_k5_2';
simu_name = 'phi2_m1_vr01_k2_2';
% simu_name = 'phi3_m2_vr001_k3_2'; 
% simu_name = 'phi4_m2_vr001_k5_3';  
% simu_name = 'phi5_m1_vr01_k5_2'; 
% simu_name = 'cars.mdl';
population_size = 50;
generations = 100;
F = 0.9;
% F = 1.2;
CR = 0.8;
input_dims = 1;
output_dims = 1;
% select dims in input
select_dims = 1;
lb = 0;
ub = 1;
% stl_req = '(alw_[0,24](b_1[t]<20))';
stl_req = 'alw_[0,18] (b_1[t] > 90 or (ev_[0,6] (b_1[t] < 50)))';
% stl_req = '(alw_[6, 12] (b_1[t] <= 10)) or (alw_[18,24] (b_1[t] > -10))';
% stl_req = 'alw_[0,19]((alw_[0,5](b_1[t] <= 20) or (ev_[0,5](b_2[t] >= 40))))';
% stl_req = 'alw_[0,19]((alw_[0,5](b_1[t] <= 20) or (ev_[0,5](b_2[t] >= 40))))';
% stl_req = 'alw_[0,65](ev_[0,30](alw_[0,5](b_5[t]-b_4[t]>=8)))';
% stl_req = '(alw_[0,50](b_2[t]-b_1[t]>7.5)) and (alw_[0,50](b_3[t]-b_2[t]>7.5)) and (alw_[0,50](b_4[t]-b_3[t]>7.5)) and (alw_[0,50](b_5[t]-b_4[t]>7.5))';


de = DE(max_time, time_step, simu_name, population_size, generations, ...
                F, CR, input_dims, output_dims, ...
                select_dims, lb, ub, stl_req);
% de.cp = 21; % number of control points
de.cp = 5;
de.run_DE(true);

traj = de.next_traj(de.cp); % Get next trajectory
