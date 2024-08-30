
kf = modelF16();
% kf.spec = specification(halfspace([0;0;0;0;0;0;0;0;0;0;0;1;0;0;0;0],0),'unsafeSet');
plot_vars = [12,13];
kf.verb=0;
kf.runs=50;
kf.nResets=5;
kf.maxSims=1000;

disp("--------------------------------------------------------")
rng(0)
kf.reach.split=false;
kfSolns = falsify(kf);
printInfo(kfSolns)

disp("--------------------------------------------------------")
rng(0)
kf.reach.split=true;
kfSolns = falsify(kf);
printInfo(kfSolns)
