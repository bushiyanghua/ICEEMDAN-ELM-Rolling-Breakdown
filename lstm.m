clear
close  all
%% 数据读取
%geshu=5557;%训练集的个数     ICEEMDAN-CWOA-LSTM的预测下： 1步5521   3步5519   6步5516      未分解情况下 1步5521    3步5519  6步5516                                  
%读取数据                      玉米  1步5559 3步5557 6步5554
forecastdata=[]
rng(501)
for i=1
  %shuru=xlsread('一步输入.xlsx',i);      %'未分解多步输入.xlsx',2  计算不同步数下的lstm        六步输入eemd                 % x步输入-m 代表不同步预测下的不同IMF '三步输入.xlsx',10       '未分解多步输入.xlsx',3
  %shuchu=xlsread('一步输出.xlsx',i);                  %'未分解多步输出.xlsx',2  计算不同步数下的lstm                         %x步输出-m 代表不同步预测下的不同IMF '三步输出.xlsx',10        '未分解多步输出.xlsx',3
  price=xlsread('3.小麦期货建模数据.xlsx',1);   %3.小麦期货建模数据   3.预测建模
  shuchu=price(21:end);
 
  price_shuru1=price(20:end-1);
  price_shuru2=price(19:end-2);
  price_shuru3=price(18:end-3);
  price_shuru4=price(17:end-4);
  price_shuru5=price(16:end-5); 
  price_shuru6=price(15:end-6); 
  price_shuru7=price(14:end-7); 
  price_shuru8=price(13:end-8); 
  price_shuru9=price(12:end-9); 
  price_shuru10=price(11:end-10); 
  price_shuru11=price(10:end-11); 
  price_shuru12=price(9:end-12); 
  price_shuru13=price(8:end-13); 
  price_shuru14=price(7:end-14); 
  price_shuru15=price(6:end-15); 
  price_shuru16=price(5:end-16); 
  price_shuru17=price(4:end-17); 
  price_shuru18=price(3:end-18); 
  price_shuru19=price(2:end-19); 
  price_shuru20=price(1:end-20);  
  
  
  %shuru=[price_shuru1,price_shuru2,price_shuru3,price_shuru4,price_shuru5,price_shuru6,price_shuru7,price_shuru8,price_shuru9,price_shuru10,price_shuru11,price_shuru12,price_shuru13,price_shuru14,price_shuru15,price_shuru16,price_shuru17,price_shuru18,price_shuru19,price_shuru20]
  shuru=[price_shuru1,price_shuru2,price_shuru3,price_shuru4,price_shuru5,price_shuru6]
  
  geshu=length(shuchu)-251
  
  
  %nn = randperm(size(shuru,1));%随机排序  %1行 248列
  nn=1:size(shuru,1);%正常排序

  input_train =shuru(nn(1:geshu),:);                             %训练集x     %200行 420列
  input_train=input_train';  %转置                               %训练集x转置  %420行 200列

  output_train=shuchu(nn(1:geshu),:);                            %训练集y     %200行 1列
  output_train=output_train';                                    %训练集y转置

  input_test =shuru(nn((geshu+1):end),:);                        %测试集x
  input_test=input_test';
  output_test=shuchu(nn((geshu+1):end),:);                       %测试集y
  output_test=output_test';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

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

  Leader_pos = [51,0.01]    %5 0.25    50，0.01   55,0.01

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


  forecastdata=[forecastdata;test_simu1]

end














