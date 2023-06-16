clear all
clc

shuru=xlsread('5.修剪高频后预测建模.xlsx',3);  %   3.预测建模_特征工程    5.修剪高频后预测建模  (这个数据集对原数据进行了删减)
%以下为插入其他控制变量                                                  %其中sheet=1、2、3、4分别对应的不同修剪力度

%kz1=xlsread('3.预测建模_特征工程.xlsx',2);                                   %前一日收盘价绝对波动
%kz2=xlsread('3.预测建模_特征工程.xlsx',3);                               %厄尔尼诺指数
%kz3=xlsread('3.预测建模_特征工程.xlsx',4);                                %花旗十国集团经济意外指数
%kz4=xlsread('3.预测建模_特征工程.xlsx',5);                                %美国每日经济不确定指数
%kz4=xlsread('3.预测建模.xlsx',5);                                             %市盈率
%划分训练集和测试集
nn=1;       %设置训练集个数 252
n=length(shuru)-nn ; 
%训练集
x_train = shuru(1:n,:)';            
%测试集-251个样本
x_test = shuru(n+1:n+nn,:)';
testdata=[];
testdata_imf=zeros(252,11);
traindata=[]


%n=length(kz1)-nn ;                                                          %此处的n比第12行的n是要大1的（从excel表里清晰可见）
                                                                            %因为这些控制变量都是前一期的数值，要预测明天，今天已经有固定的值了。
%kz1_train=kz1(1:n,:)'   ;                                                         %kz1
%kz1_1=kz1(n+1:n+nn,:)'  ;   %kz1
%kz2_train=kz2(1:n,:)'   ;                                                         %kz2
%kz2_1=kz2(n+1:n+nn,:)'  ;                                                         %kz2
%kz3_train=kz3(1:n,:)'   ;                                                         %kz3
%kz3_1=kz3(n+1:n+nn,:)'  ;                                                         %kz3
%kz4_train=kz4(1:n,:)'   ;                                                         %kz4
%kz4_1=kz4(n+1:n+nn,:)'  ;                                                         %kz4

for ii=1:nn
            
    x1=x_test(ii)
    x_train=[x_train,x1]   %窗口滚动之增加当日收盘价新值
    
    %k1=kz1_1(ii)                                                              %kz1
    %kz1_train=[kz1_train,k1]                                                  %kz1
    %k2=kz2_1(ii)                                                              %kz2
    %kz2_train=[kz2_train,k2]                                                  %kz2
    %k3=kz3_1(ii)                                                              %kz3
    %kz3_train=[kz3_train,k3]   
    %k4=kz4_1(ii)                                                              %kz4
    %kz4_train=[kz4_train,k4]                                                  %kz4
    

    %N=length(x_train);  %原油5777 玉米
    %t=1/N:1/N:1;

    %figure(1)
    %subplot(1,1,1);
    %plot(t,x_train);xlabel('时间');ylabel('价格');title('WTI原油期货价格波动序列');

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    NR=100;
    Nstd=0.05;
    MaxIter=5;
    SNRFlag=2;
 
    x=x_train(:)';                      % x 1*2048
    desvio_x=std(x);
    x=x/desvio_x;
    
    medias=zeros(size(x));
    modes=zeros(size(x));
    temp=zeros(size(x));
    aux=zeros(size(x));
    iter=zeros(NR,round(log2(length(x))+5));



    for i=1:NR
        rng(i+500) %设置随机数种子
        white_noise{i}=randn(size(x));%creates the noise realizations   %1*2048
    end;

    for i=1:NR
        modes_white_noise{i}=emd(white_noise{i});%calculates the modes of white gaussian noise 50个序列 又拆分成9列imf值
    end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%此处大改
    for i=1:NR   %calculates the first mode
        mi=modes_white_noise{i}(:,1); %此处原为(1,:)，所以报错，仔细探究原理后发现源代码此处写错，故修改之
        xi=x+Nstd*mi(:)'/std(mi);  %第二处错误是向量问题，修改后mi是2048*1的向量，mi
        [temp, o1, it1]=emd(xi,'MaxNumIMF',MaxIter,'MaxNumExtrema',1);
        temp=temp(:,1);               %temp为2048行6列  ；x为1行2048列;此处原为(1,:)，所以报错
    
        aux=aux+(xi-temp(:)')/NR;      %xi为1*2048 ，temp为2048*1 # 所有循环后此处得到的aux是iceemdan中，50个序列的总局部均值
    %iter(i,1)=it1;
    end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    modes= x-aux; %saves the first mode  ；aux为第一次的总局均值 ；modes是真正意义上的imf1
    medias = aux; %medias是第二次处理的“原序列”，将aux的值赋予之
    k=1;
    aux=zeros(size(x)); %将aux归零
    es_imf = min(size(emd(medias(end,:),'MaxNumIMF',MaxIter,'MaxNumExtrema',1)));  %返回在50个序列中imf最少的那个序列的imf个数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%上面这个end是干啥的？有什么特殊目的吗，为什么不用1？


    while es_imf>1 %calculates the rest of the modes  ；如果“原序列”仍能被emd分解出超过两个及以上imf；
 % 此处大胆修改，原来是>1,后来被我放松到2,减少了imf个数
        for i=1:NR
            tamanio=size(modes_white_noise{i});   %tomanio[2048,9] 白噪声序列的规格
            if tamanio(2)>=k+1        % 如果白噪声被emd分解出的emd大于等于k+1；即如果k+1小于等于9的时候运行  因为有的白噪声只有5个imf分量，此时若原序列已经计算到imf6，则没有对应的白噪声imf6与之相加，因此这个x序列就要被抛弃
                noise=modes_white_noise{i}(:,k+1);  %修改 ； 则提取出白噪声序列的第2项，即，求imf2的时候加的是白噪声的imf2项
                if SNRFlag == 2
                    noise=noise/std(noise); %adjust the std of the noise
                end;          
                noise=Nstd*noise; % 形成新一轮的白噪声
                noise=noise(:)' ;%这一条是我增加的，目的是将向量medias和noise行列对齐
                try
                    [temp,o,it]=emd(medias(end,:)+noise,'MaxNumIMF',MaxIter,'MaxNumExtrema',1);
    %大胆修改输入emd的x，原来是medias(end,:)+std(medias(end,:))*noise，
                catch    
                    it=0; disp('catch 1 '); disp(num2str(k))
                    temp=emd(medias(end,:)+noise,'MaxNumIMF',MaxIter,'MaxNumExtrema',1);
    %大胆修改输入emd的x，原来是medias(end,:)+std(medias(end,:))*noise，
                end;
                temp=temp(:,1);      %此处修改，行列对调 temp 2048行，6列 ;此处提取出来了emd分解后新的imf1
   %%%此处有一个冒险的改动，将end改成了1
            else
                try
                    [temp, o, it]=emd(medias(end,:),'MaxNumIMF',MaxIter,'MaxNumExtrema',1);
                catch
                    temp=emd(medias(end,:),'MaxNumIMF',MaxIter,'MaxNumExtrema',1);
                    it=0; disp('catch 2 sin ruido')
                end;
                temp=temp(:,1);   %此处修改，行列对调 temp 2048行，6列;此处提取出来了emd分解后的趋势项
   %%%此处有一个冒险的改动，将end改成了1
            end;
            aux=aux+(medias(end,:)-temp(:)')/NR;        %aux是局部均值的平均——残差
   %%%此处有大修改，原来是 aux=aux+temp/NR ，存有重大原理错误
    %iter(i,k+1)=it;    
        end;
        modes=[modes;medias(end,:)-aux]; %原残差-局部均值的平均=imf2   modes储存的是imf
        medias = [medias;aux]; %变成2行，2048列                       medias储存的是每一次的残差
        aux=zeros(size(x));
        k=k+1;
        es_imf = min(size(emd(medias(end,:),'MaxNumIMF',MaxIter,'MaxNumExtrema',1)));
    end;
    
    modes = [modes;medias(end,:)]; %加上最后的残差项 即所有imf＋残差
    modes=modes*desvio_x;                          

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%上为分解，下为预测


    geshu=size(modes);    %查看本征模函数的个数和列数
    number_imf=geshu(1) ; %保存本征模函数个数
    b=unifrnd(0,1:number_imf,[1 number_imf]) ; %(生成x个 1-x的随机数)
    
    modesf=[] 
    modesf=[modes,b'];
    forecastdata=[];
    train=[]
    
%以下为重构高频、低频和趋势项
    mmmm_gaopin=modesf(1,:)+modesf(2,:)+modesf(3,:)             %此处为高频优化部分    %modesf(1,:)+  +modesf(4,:)   modesf(1,:)+
    mmmm_zhongpin=modesf(4,:)
    for i=5:number_imf-2              %number_imf-1
        mmmm_zhongpin=mmmm_zhongpin+modesf(i,:);
    end;
   
    mmmm_dipin=modesf((number_imf-1),:)+modesf(number_imf,:)
    
    
    %for i=8:number_imf            %number_imf-1
        %mmmm_dipin=mmmm_dipin+modesf(i,:);
    %end;
    
    
    
    
          
    modesf=[] 
    modesf=[mmmm_gaopin',mmmm_zhongpin',mmmm_dipin']
    modesf=modesf'
    
    for i=1:3
        
        
        
        IMF=modesf(i,:); %此处取出IMF1，接下来构造IMF1的输出和输入集

        IMF_shuchu=IMF(21:end);
        octane=IMF_shuchu';

        IMF_shuru1=IMF(20:end-1);
        IMF_shuru2=IMF(19:end-2);
        IMF_shuru3=IMF(18:end-3);
        IMF_shuru4=IMF(17:end-4);
        IMF_shuru5=IMF(16:end-5);
        IMF_shuru6=IMF(15:end-6);
        IMF_shuru7=IMF(14:end-7);
        IMF_shuru8=IMF(13:end-8);
        IMF_shuru9=IMF(12:end-9);
        IMF_shuru10=IMF(11:end-10);
        IMF_shuru11=IMF(10:end-11);
        IMF_shuru12=IMF(9:end-12);
        IMF_shuru13=IMF(8:end-13);
        IMF_shuru14=IMF(7:end-14);
        IMF_shuru15=IMF(6:end-15);
        IMF_shuru16=IMF(5:end-16);
        IMF_shuru17=IMF(4:end-17);
        IMF_shuru18=IMF(3:end-18);
        IMF_shuru19=IMF(2:end-19);
        IMF_shuru20=IMF(1:end-20);
       
        
                
        %kz1=kz1(5:end-1);
        %kz2=kz2(5:end-1);
        %kz3=kz3(5:end-1);
        

        %%此处是同期11，是因为在excel中已经进行了错位，无需再次错位
        %kz1_train1=kz1_train(21:end);
        %kz2_train1=kz2_train(21:end);                                                     %kz2
        %kz3_train1=kz3_train(21:end);                                                     %kz3
        %kz4_train1=kz4_train(21:end);                                                     %kz4
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%此处为特征工程
        if i == 1
            NIR=[IMF_shuru1',IMF_shuru2',IMF_shuru3',IMF_shuru4',IMF_shuru5',IMF_shuru6'];    %kz  kz1_train1',kz4_train1',   ,IMF_shuru2',IMF_shuru3'
            num=20;               %此处设置elm在不同高中低频率下的参数
        end;
   
        if i == 2
            NIR=[IMF_shuru1',IMF_shuru2',IMF_shuru3',IMF_shuru4',IMF_shuru5',IMF_shuru6'];    %kz  kz1_train1',kz4_train1',
            num=20;  %,IMF_shuru12',IMF_shuru15',IMF_shuru18',IMF_shuru20'
        end;
        
        if i == 3
            NIR=[IMF_shuru1',IMF_shuru2',IMF_shuru3',IMF_shuru4',IMF_shuru5',IMF_shuru6'];    %kz  kz1_train1',kz4_train1',
            num=20
        end;
        
        %if i == 4
            %NIR=[IMF_shuru1',IMF_shuru2',IMF_shuru3',IMF_shuru4',IMF_shuru5',IMF_shuru6',IMF_shuru7',IMF_shuru8',IMF_shuru9',IMF_shuru10'];    %   ,kz2_train1',kz3_train1',kz4_train1'
            %num=20
        %end;
        
        
        
%kz1_train1',kz1_train2',kz4_train1',kz4_train2',
        nn=1           %设置训练集个数
        n=length(NIR)-nn % 

%%2.随机产生训练集和测试集
%temp = randperm(size(NIR,1));
%训练集-      
        P_train = NIR(1:n,:)';            
        T_train = octane(1:n,:)';

%测试集-251个样本
        P_test = NIR(n+1:n+nn,:)';
        T_test = octane(n+1:n+nn,:)';
        N = size(P_test,2);

%%3.归一化数据
% 3.1. 训练集
        [Pn_train,inputps] = mapminmax(P_train);
        Pn_test = mapminmax('apply',P_test,inputps);
% 3.2. 测试集
        [Tn_train,outputps] = mapminmax(T_train);
        Tn_test = mapminmax('apply',T_test,outputps);

%%4.ELM训练
        [IW1,B1,H1,TF1,TYPE1] = elmtrain(Pn_train,Tn_train,num,'sig',0);
%%5.ELM仿真测试
        tn_sim01 = elmpredict(Pn_test,IW1,B1,H1,TF1,TYPE1);
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
        %figure(1);
        %plot(1:N,T_test,'r-*',1:N,T_sim,'b:o');
        %grid on 
        %legend('真实值','预测值')
        %xlabel('样本编号')
        %ylabel('值')
        forecastdata=[forecastdata;T_sim]
        testdata_imf(ii,i)=T_sim;
        
        %train=[train;T_train]  %记录样本内模型拟合值（分频率），用于计算MSE
        
    end;
    
    
    testdata=[testdata;sum(forecastdata)]
    %traindata=[traindata;sum(train)]  %记录样本内模型拟合值，用于计算MSE


end;

testdata=testdata'
  
%E = mse(traindata - x_train(1,21:end));  %用于计算MSE

    x=(traindata - x_train(1,21:end))^2
  
  