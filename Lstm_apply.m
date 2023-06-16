
%% 加载数据
load('imfr_iceemdan.mat')
data=modes(7,:)  %imf1 1*2541
figure()
plot(data,LineWidth=2)


%% 数据拆分及标准化
numTimeStepsTrain = floor(numel(data)-7);     %六步应当减7
dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end); 

mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig; 

%要预测序列的未来时间步长的值，请将响应指定为值移位一个时间步长的训练序列。
%也就是说，在输入序列的每个时间步长，LSTM网络学习预测下一个时间步长的值。
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);  % 用六步前的情况预测现在 ，这样岂不是没用到最近五天的信息？？？？


%% 定义网络结构
% 定义网络结构
layers = [
    sequenceInputLayer(1,"Name","input")
    lstmLayer(128,"Name","lstm")
%     dropoutLayer(0.2,"Name","drop")
    fullyConnectedLayer(1,"Name","fc")
    regressionLayer];
% 定义训练参数
options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress')

%% 训练网络
% 训练网络
net = trainNetwork(XTrain,YTrain,layers,options); 

%% 预测(每一次使用上一次的预测值）主要用这个预测
% 测试集归一化
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);

% 预测
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));   %训练集的最后一个y 进入测试集第一个y的预测

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end 

YPred = sig*YPred + mu;          %作者：zi_hu https://www.bilibili.com/read/cv15809353 出处：bilibili


% 绘图
figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'k.-'),hold on
plot(idx,data(numTimeStepsTrain:end-1),'r'),hold on
hold off
xlabel("Day")
ylabel("P")
title("Forecast")
legend(["Observed" "Forecast"]) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 预测(每一次使用上一次的真实值）或许可以用于稳定性检验 此模块暂时不用
net = resetState(net);
net = predictAndUpdateState(net,XTrain); 

% 重置网络状态
net = resetState(net);
net = predictAndUpdateState(net,XTrain);
%

YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end

YPred = sig*YPred + mu;

% 绘图
figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'k.-'),hold on
plot(idx,data(numTimeStepsTrain:end-1),'r'),hold on
hold off
xlabel("Day")
ylabel("P")
title("Forecast")
legend(["Observed" "Forecast"]) 

%%








