clear all
clc
forecastdata=[]
rng(501)
for i=1
    
    price=xlsread('3.С���ڻ���ģ����.xlsx',1);        %   3.Ԥ�⽨ģ.xlsx     xlsread('3.С���ڻ���ģ����.xlsx',1); 
    %kz1_train=xlsread('3.Ԥ�⽨ģ.xlsx',2);
    %kz4_train=xlsread('3.Ԥ�⽨ģ.xlsx',3);
    
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


%%1.��������
%load datainput1.mat   %ICEEMDAN_IMF1INPUT.mat   %        %  1��Ԥ��datainput1    3��Ԥ��datainput3      6��Ԥ��datainput6
%load dataoutput1.mat  %ICEEMDAN_IMF1OUTPUT.mat  %
    n=length(NIR)-251 % 1��Ԥ��5521       3��Ԥ��5519       6��Ԥ��5516
%%2.�������ѵ�����Ͳ��Լ�
%temp = randperm(size(NIR,1));
%ѵ����-      
    P_train = NIR(1:n,:)';            
    T_train = octane(1:n,:)';

%���Լ�-251������
    P_test = NIR(n+1:n+251,:)';
    T_test = octane(n+1:n+251,:)';
    N = size(P_test,2);

%%3.��һ������
% 3.1. ѵ����
    [Pn_train,inputps] = mapminmax(P_train);
    Pn_test = mapminmax('apply',P_test,inputps);
% 3.2. ���Լ�
    [Tn_train,outputps] = mapminmax(T_train);
    Tn_test = mapminmax('apply',T_test,outputps);

%%4.ELMѵ��
    [IW1,B1,H1,TF1,TYPE1] = elmtrain(Pn_train,Tn_train,20,'sig',0);
%[IW2,B2,H2,TF2,TYPE2] = elmtrain(H1,Tn_train,20,'sig',0);
%[IW3,B3,H3,TF3,TYPE3] = elmtrain(H2,Tn_train,30,'sig',0);
%LW = pinv(H1') * Tn_train';
%%5.ELM�������
    tn_sim01 = elmpredict(Pn_test,IW1,B1,H1,TF1,TYPE1);
%tn_sim02 = elmpredict(tn_sim01,IW2,B2,TF2,TYPE2);
%tn_sim03 = elmpredict(tn_sim02,IW3,B3,TF3,TYPE3);
%����ģ�����
%tn_sim = (tn_sim01' * LW)';
%5.1. ����һ��
    T_sim = mapminmax('reverse',tn_sim01,outputps);

%%6.����Ա�
    result = [T_test' T_sim'];
%6.1.�������
    E = mse(T_sim - T_test);

%6.2 ���ϵ��
    N = length(T_test);
    %R2 = (N*sum(T_sim.*T_test)-sum(T_sim)*sum(T_test))^2/((N*sum((T_sim).^2)-(sum(T_sim))^2)*(N*sum((T_test).^2)-(sum(T_test))^2));

%%7.��ͼ
    figure(1);
    plot(1:N,T_test,'r-*',1:N,T_sim,'b:o');
    grid on 
    legend('��ʵֵ','Ԥ��ֵ')
    xlabel('�������')
    ylabel('ֵ')
    %string = {'Ԥ�����Ա�(ELM)';['(mse = ' num2str(E) ' R^2 = ' num2str(R2) ')']};
    %title(string)
    %forecastdata{i}=T_sim
    forecastdata=[forecastdata;T_sim]
   
    
end
