clear
close  all
%% 数据读取
geshu=200;%训练集的个数                                                    
%读取数据
shuru=xlsread('数据的输入.xlsx');                                           % 248行 420列
shuchu=xlsread('数据的输出.xlsx');                                          %248行 1列
nn = randperm(size(shuru,1));%随机排序                                      %1行 248列
% nn=1:size(shuru,1);%正常排序

input_train =shuru(nn(1:geshu),:);                             %训练集x     %200行 420列
input_train=input_train';  %转置                               %训练集x转置  %420行 200列
output_train=shuchu(nn(1:geshu),:);                            %训练集y     %200行 1列
output_train=output_train';                                    %训练集y转置

input_test =shuru(nn((geshu+1):end),:);                        %测试集x
input_test=input_test';
output_test=shuchu(nn((geshu+1):end),:);                       %测试集y
output_test=output_test';


%样本输入输出数据归一化
[aa,bb]=mapminmax([input_train input_test]);
[cc,dd]=mapminmax([output_train output_test]);
global inputn outputn shuru_num shuchu_num

[inputn,inputps]=mapminmax('apply',input_train,bb);                        %420行 200列
[outputn,outputps]=mapminmax('apply',output_train,dd);                     %1行   200列    
shuru_num = size(input_train,1); % 输入维度                                %维度420
shuchu_num = 1;  % 输出维度
%%
% 参数初始化
dim=2;
Max_iteration=20;   % 进化次数  
pop=5;  %种群规模
lb = [300, 0.15];
ub = [50, 0.01];
fobj = @(x) fun(x);
[Leader_score,Leader_pos,Convergence_curve]=WOA(pop,Max_iteration,lb,ub,dim,fobj); %开始优化

g1=Leader_pos;
zhongjian1_num = round(g1(1));  
xue = g1(2);
%% 模型建立与训练
layers = [ ...
    sequenceInputLayer(shuru_num)    % 输入层
    lstmLayer(zhongjian1_num)        % LSTM层
    fullyConnectedLayer(shuchu_num)  % 全连接层
    regressionLayer];
 
options = trainingOptions('adam', ...   % 梯度下降
    'MaxEpochs',50, ...                % 最大迭代次数
    'GradientThreshold',1, ...         % 梯度阈值 
    'InitialLearnRate',xue,...
    'Verbose',0, ...
    'Plots','training-progress');            % 学习率
%% 训练LSTM
net = trainNetwork(inputn,outputn,layers,options);
%% 预测
net = resetState(net);% 网络的更新状态可能对分类产生了负面影响。重置网络状态并再次预测序列。
[~,Ytrain]= predictAndUpdateState(net,inputn);
test_simu=mapminmax('reverse',Ytrain,dd);%反归一化
rmse = sqrt(mean((test_simu-output_train).^2));   % 训练集
%测试集样本输入输出数据归一化
inputn_test=mapminmax('apply',input_test,bb);
[net,an]= predictAndUpdateState(net,inputn_test);
test_simu1=mapminmax('reverse',an,dd);%反归一化
error1=test_simu1-output_test;%测试集预测-真实
%计算均方根误差 (RMSE)。
rmse1 = sqrt(mean((test_simu1-output_test).^2));  % 测试集
%% 画图

%将预测值与测试数据进行比较。
figure
plot(output_train)
hold on
plot(test_simu,'.-')
hold off
legend(["真实值" "预测值"])
xlabel("样本")
title("训练集")


figure
plot(output_test)
hold on
plot(test_simu1,'.-')
hold off
legend(["真实值" "预测值"])
xlabel("样本")
title("测试集")


 % 真实数据，行数代表特征数，列数代表样本数output_test = output_test;

T_sim_optimized = test_simu1;  % 仿真数据

num=size(output_test,2);%统计样本总数
error=T_sim_optimized-output_test;  %计算误差
mae=sum(abs(error))/num; %计算平均绝对误差
me=sum((error))/num; %计算平均绝对误差
mse=sum(error.*error)/num;  %计算均方误差
rmse=sqrt(mse);     %计算均方误差根
% R2=r*r;
tn_sim = T_sim_optimized';
tn_test =output_test';
N = size(tn_test,1);
R2=(N*sum(tn_sim.*tn_test)-sum(tn_sim)*sum(tn_test))^2/((N*sum((tn_sim).^2)-(sum(tn_sim))^2)*(N*sum((tn_test).^2)-(sum(tn_test))^2)); 

disp(' ')
disp('----------------------------------------------------------')

disp(['平均绝对误差mae为：            ',num2str(mae)])
disp(['平均误差me为：            ',num2str(me)])
disp(['均方误差根rmse为：             ',num2str(rmse)])
disp(['相关系数R2为：                ' ,num2str(R2)])




















