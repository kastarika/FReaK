% initialize seeds
rng(1234)
pyrunfile("seed.py")

kfModel = modelSynth2();
plotVars = [1];

ose = t2();
kfModel.ose = ose;
disp(class(kfModel.ose));
% x = stl('x',3);
% eq = implies(globally(x(2)<3000,interval(0,30)),globally(x(1)<35,interval(0,4)));
% kfModel.spec = specification(eq,'logic');

kfModel.verb=2;

[kfSolns,allDatas] = falsify(kfModel);
kfSoln=kfSolns{1};
allData=allDatas{1};

if kfSoln.falsified
    disp(['simulations required: ',num2str(kfSoln.sims)])
    visualizeFalsification(kfSoln.best.x,kfSoln.best.t,kfSoln.best.spec,plotVars,'Speed','Angular velocity')
else
    disp("No falsifiying trace found")
end

% visualizeTrain(allData,kfModel.ak.dt,plotVars,'Speed','Angular velocity')
%settings for figure
% figure_settings(gcf);
% export_fig training.pdf