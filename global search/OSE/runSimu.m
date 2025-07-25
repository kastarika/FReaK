% inputs simulink file name as string
function [tout, yout] = runSimu(simu_name, T, ~, u)
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
