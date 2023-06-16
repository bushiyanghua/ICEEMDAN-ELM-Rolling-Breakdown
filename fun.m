function error = fun(pop)
global inputn outputn shuru_num shuchu_num
tic

pop(1)=round(pop(1));
layers = [ ...
    sequenceInputLayer(shuru_num)
    lstmLayer(pop(1))
    fullyConnectedLayer(shuchu_num)
    regressionLayer];
options = trainingOptions('adam', ...  % �ݶ��½�
    'MaxEpochs',50, ...                % ����������
     'GradientThreshold',1, ...         % �ݶ���ֵ 
    'InitialLearnRate',pop(2));

% ����ѵ����=ѵ������ѡȡ80%����ѵ����20%����ѵ������
n = randperm(size(inputn,2));%���ѡȡ
xun_n = round(size(inputn,2)*0.8);

xunx = inputn(:,n(1:xun_n));
xuny = outputn(:,n(1:xun_n));  

cex = inputn(:,n((xun_n+1):end));
cey = outputn(:,n((xun_n+1):end)); 

% ѵ��LSTM
net = trainNetwork(xunx,xuny,layers,options);
% Ԥ��
net = resetState(net);% ����ĸ���״̬���ܶԷ�������˸���Ӱ�졣��������״̬���ٴ�Ԥ�����С�


[~,Ytrain]= predictAndUpdateState(net,cex);
cg = mse(Ytrain,cey);
 toc
disp('-------------------------')
error = cg;
end