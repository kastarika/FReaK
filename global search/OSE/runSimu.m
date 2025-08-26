% inputs simulink file name as string
function [tout, yout] = runSimu(simu_name, T, ~, u)
    % ts = u(:,1);
    % us = u(:,2:end);
    % 
    % tin = 0:0.01:T;
    % xin = interp1(ts, us, tin, 'previous');
    % disp(size(ts))
    % disp(size(us))
    % disp(size(tin))
    % disp(size(xin))
    % u = [tin' xin];
    % u_ts=0.001;
    % 
    % assignin('base','u',u);
    % assignin('base','T',T);
    % assignin('base','u_ts',u_ts);

    % evalin("base","init_neural;")
    % ts = u(:,1);
    % us = u(:,2:end);
    % 
    % tin = 0:0.01:T;
    % xin = interp1(ts, us, tin, 'previous');
    % u = [tin' xin'];
    % 
    % assignin('base','u',u);
    % assignin('base','T',T);
    
    % assignin('base','u',u);
    % assignin('base','T',T);
    

    % evalin("base","init_powertrain;")
    % ts = u(:,1);
    % us = u(:,2:end);
    % 
    % tin = 0:0.01:T;
    % xin = interp1(ts, us, tin, 'previous');
    % u = [tin' xin];
    
    assignin('base','u',u);
    assignin('base','T',T);
    
    result = sim(simu_name, ...
        'StopTime', 'T', ...
        'LoadExternalInput', 'on', 'ExternalInput', 'u', ...
        'SaveTime', 'on', 'TimeSaveName', 'tout', ...
        'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Array');
    tout = result.tout;
    yout = result.yout;
end
