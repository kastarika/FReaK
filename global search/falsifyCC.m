% initialize seeds
rng(1234)
pyrunfile("seed.py")

model = @modelCars2;
kfModel = model();
plotVars = [1];

% ose = tCC();
% kfModel.ose = ose;
% disp(class(kfModel.ose));
x = stl('x',5);
eq = globally(finally(globally(x(5)-x(4)>=8,interval(0,5)),interval(0,30)),interval(0,65));
kfModel.spec = specification(eq,'logic');
kfModel.resetStrat=0;
% kfModel.trainStrat=2;
kfModel.verb=-1;
% kfModel.reach.on=false;

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