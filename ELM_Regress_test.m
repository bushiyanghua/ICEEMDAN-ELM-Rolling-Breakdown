clear all
clc
forecastdata=[]
rng(501)
for i=1
    
    price=xlsread('3.小麦期货建模数据.xlsx',1);        %   3.预测建模.xlsx     xlsread('3.小麦期货建模数据.xlsx',1); 
    %kz1_train=xlsread('3.预测建模.xlsx',2);
    %kz4_train=xlsread('3.预测建模.xlsx',3);
    
    price_shuchu=price(21:end);
    octane=price_shuchu;

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
    %kz1_train1=kz1_train(21:end-1);                                                     %kz1
    %kz4_train1=kz4_train(21:end-1); 
        
    %NIR=[price_shuru1,price_shuru2,price_shuru3,price_shuru4,price_shuru5,price_shuru6,price_shuru7,price_shuru8,price_shuru9,price_shuru10] %  kz1_train1,kz4_train1   ,price_shuru4,price_shuru5  price_shuru12,price_shuru17
    %NIR=[price_shuru1,price_shuru2,price_shuru3,price_shuru4,price_shuru5,price_shuru6,price_shuru7,price_shuru8,price_shuru9,price_shuru10,price_shuru11,price_shuru12,price_shuru13,price_shuru14,price_shuru15,price_shuru16,price_shuru17,price_shuru18,price_shuru19,price_shuru20];    %kz kz1_train1,kz4_train1,
    NIR=[price_shuru1,price_shuru2,price_shuru3,price_shuru4,price_shuru5,price_shuru6]
    %,price_shuru6,price_shuru7,price_shuru8,price_shuru9,price_shuru10


%%1.导入数据
%load datainput1.mat   %ICEEMDAN_IMF1INPUT.mat   %        %  1步预测datainput1    3步预测datainput3      6步预测datainput6
%load dataoutput1.mat  %ICEEMDAN_IMF1OUTPUT.mat  %
    n=length(NIR)-251 % 1步预测5521       3步预测5519       6步预测5516
%%2.随机产生训练集和测试集
%temp = randperm(size(NIR,1));
%训练集-      
    P_train = NIR(1:n,:)';            
    T_train = octane(1:n,:)';

%测试集-251个样本
    P_test = NIR(n+1:n+251,:)';
    T_test = octane(n+1:n+251,:)';
    N = size(P_test,2);

%%3.归一化数据
% 3.1. 训练集
    [Pn_train,inputps] = mapminmax(P_train);
    Pn_test = mapminmax('apply',P_test,inputps);
% 3.2. 测试集
    [Tn_train,outputps] = mapminmax(T_train);
    Tn_test = mapminmax('apply',T_test,outputps);

%%4.ELM训练
    [IW1,B1,H1,TF1,TYPE1] = elmtrain(Pn_train,Tn_train,20,'sig',0);
%[IW2,B2,H2,TF2,TYPE2] = elmtrain(H1,Tn_train,20,'sig',0);
%[IW3,B3,H3,TF3,TYPE3] = elmtrain(H2,Tn_train,30,'sig',0);
%LW = pinv(H1') * Tn_train';
%%5.ELM仿真测试
    tn_sim01 = elmpredict(Pn_test,IW1,B1,H1,TF1,TYPE1);
%tn_sim02 = elmpredict(tn_sim01,IW2,B2,TF2,TYPE2);
%tn_sim03 = elmpredict(tn_sim02,IW3,B3,TF3,TYPE3);
%计算模拟输出
%tn_sim = (tn_sim01' * LW)';
%5.1. 反归一化
    T_sim = mapminmax('reverse',tn_sim01,outputps);

%%6.结果对比
    result = [T_test' T_sim'];
%6.1.均方误差
    E = mse(T_sim - T_test);

%6.2 相关系数
    N = length(T_test);
    %R2 = (N*sum(T_sim.*T_test)-sum(T_sim)*sum(T_test))^2/((N*sum((T_sim).^2)-(sum(T_sim))^2)*(N*sum((T_test).^2)-(sum(T_test))^2));

%%7.成图
    figure(1);
    plot(1:N,T_test,'r-*',1:N,T_sim,'b:o');
    grid on 
    legend('真实值','预测值')
    xlabel('样本编号')
    ylabel('值')
    %string = {'预测结果对比(ELM)';['(mse = ' num2str(E) ' R^2 = ' num2str(R2) ')']};
    %title(string)
    %forecastdata{i}=T_sim
    forecastdata=[forecastdata;T_sim]
   
    
end
