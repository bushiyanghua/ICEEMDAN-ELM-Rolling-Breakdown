function error = fun(pop)
global inputn outputn shuru_num shuchu_num
tic

pop(1)=round(pop(1));
layers = [ ...
    sequenceInputLayer(shuru_num)
    lstmLayer(pop(1))
    fullyConnectedLayer(shuchu_num)
    regressionLayer];
options = trainingOptions('adam', ...  % 梯度下降
    'MaxEpochs',50, ...                % 最大迭代次数
     'GradientThreshold',1, ...         % 梯度阈值 
    'InitialLearnRate',pop(2));

% 划分训练集=训练集中选取80%进行训练，20%进行训练测试
n = randperm(size(inputn,2));%随机选取
xun_n = round(size(inputn,2)*0.8);

xunx = inputn(:,n(1:xun_n));
xuny = outputn(:,n(1:xun_n));  

cex = inputn(:,n((xun_n+1):end));
cey = outputn(:,n((xun_n+1):end)); 

% 训练LSTM
net = trainNetwork(xunx,xuny,layers,options);
% 预测
net = resetState(net);% 网络的更新状态可能对分类产生了负面影响。重置网络状态并再次预测序列。


[~,Ytrain]= predictAndUpdateState(net,cex);
cg = mse(Ytrain,cey);
 toc
disp('-------------------------')
error = cg;
end