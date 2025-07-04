function [tout, yout] = run_synth_benchmark2(~, u, T)
    assignin('base','u',u);
    assignin('base','T',T);
    
    result = sim('phi2_m1_vr01_k2_2', ...
        'StopTime', 'T', ...
        'LoadExternalInput', 'on', 'ExternalInput', 'u', ...
        'SaveTime', 'on', 'TimeSaveName', 'tout', ...
        'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Array');
    tout = result.tout;
    yout = result.yout;
end
