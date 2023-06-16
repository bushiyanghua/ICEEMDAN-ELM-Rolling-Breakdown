# 设置工作空间
# 把“数据及程序”文件夹拷贝到F盘下，再用setwd设置工作空间
setwd("D:/桌面文件/中南财经政法/期刊论文/基于分解-重构策略的神经网络预测大宗商品期货价格方法")

library(openxlsx)
library(forecast)
library(TSA)

price<- read.xlsx('3.预测建模.xlsx',sheet=1)
price<-ts(price)
n=length(price) 


################################################################################ARIMAX仅用于样本内，精度虚高
shuchu=price[11:n];
price_shuru1=as.data.frame(price[10:(n-1)]);
price_shuru2=as.data.frame(price[9:(n-2)]);
price_shuru3=as.data.frame(price[8:(n-3)]);
price_shuru4=as.data.frame(price[7:(n-4)]);
price_shuru5=as.data.frame(price[6:(n-5)]); 
price_shuru6=as.data.frame(price[5:(n-6)]); 
price_shuru7=as.data.frame(price[4:(n-7)]); 
price_shuru8=as.data.frame(price[3:(n-8)]); 
price_shuru9=as.data.frame(price[2:(n-9)]); 
price_shuru10=as.data.frame(price[1:(n-10)]);  


#create data frame with 0 rows and 3 columns
df.empty <- data.frame(matrix(ncol = 10, nrow = length(price_shuru1[,1])))
#provide column names
colnames(df.empty) <- c('price_shuru1', 'price_shuru2', 'price_shuru3', 'price_shuru4', 'price_shuru5', 'price_shuru6', 'price_shuru7', 'price_shuru8', 'price_shuru9', 'price_shuru10')
df.empty$price_shuru1=price_shuru1
df.empty$price_shuru2=price_shuru2
df.empty$price_shuru3=price_shuru3
df.empty$price_shuru4=price_shuru4
df.empty$price_shuru5=price_shuru5
df.empty$price_shuru6=price_shuru6
df.empty$price_shuru7=price_shuru7
df.empty$price_shuru8=price_shuru8
df.empty$price_shuru9=price_shuru9
df.empty$price_shuru10=price_shuru10

x <- df.empty
x<-as.matrix(x)
y1 <- shuchu
y1<-ts(y1)
n=length(y1) 

######################ARIMAX建模
#x1=x[1:(n-251),1:10]
#y11=y1[1:(n-251)]
x1=x
y11=y1
model_with_var= auto.arima(y11, xreg=x1)

######################ARIMAX预测

x2=x[(n-250):n,1:10]
x2=ts(x2)

forecast4=forecast(model_with_var,h=251,xreg = x2)
mm<-forecast4$fitted[(n-250):n]

plot(y1[(n-250):n],type="l")    #plot(y1[(n-250):n],type="p",pch="*")
lines(mm, type="l", col="red") # 画出第二个时间序列y，并用蓝色标识

forecast1<-data.frame(mm)
write.xlsx(forecast1,file='C:/Users/不是杨花/Desktop/forecast1.xlsx')

################################################################################ARIMA样本外滚动预测
# 设置工作空间
# 把“数据及程序”文件夹拷贝到F盘下，再用setwd设置工作空间
setwd("D:/桌面文件/中南财经政法/期刊论文/基于分解-重构策略的神经网络预测大宗商品期货价格方法")
library(openxlsx)
library(forecast)
library(TSA)

price<- read.xlsx('3.预测建模.xlsx',sheet=1)
price<-ts(price)
n=length(price) 


price1 = price[1:(n-252)]
price2 = price[(n-251):n]
df.empty <- data.frame(matrix(ncol = 1, nrow = 251))

for (j in 1:251){
  
  price1 = append(price1,price2[j])
  fit=auto.arima(price1)
  fore=forecast::forecast(fit,h=1)
  df.empty[j,1]=fore$mean
}

write.xlsx(df.empty,file='C:/Users/不是杨花/Desktop/forecast1.xlsx')


############################################################################ SVR
# 设置工作空间
# 把“数据及程序”文件夹拷贝到F盘下，再用setwd设置工作空间
setwd("D:/桌面文件/中南财经政法/期刊论文/基于分解-重构策略的神经网络预测大宗商品期货价格方法")
library(openxlsx)

for (i in 1){
  
  dataset <- read.xlsx('SVR.xlsx',sheet=5)    #4是原油使用的数据；5是小麦使用的数据
  # Splitting the dataset into the Training set and Test set
  #install.packages('caTools')
  #library(caTools)
  #set.seed(123)
  # split = sample.split(dataset$Salary, SplitRatio = 2/3)
  # training_set = subset(dataset, split == TRUE)
  # test_set = subset(dataset, split == FALSE)
  stepahead=251
  data_trn <- head(dataset,round(length(dataset[,1]) - stepahead-1))
  data_test <- tail(dataset, stepahead)
  # Feature Scaling
  # training_set = scale(training_set)
  # test_set = scale(test_set)
  
  # Fitting SVR to the dataset
  # install.packages('e1071')
  library(e1071)
  regressor = svm(formula = Y ~ .,
                  data = data_trn,cost=0.001,    #小麦0.001
                  type = 'nu-regression',  # C-classification    eps-regression   nu-regression
                  kernel = 'linear') #linear     polynomial  radial  sigmoid
  
  # Predicting a new result
  y_pred = predict(regressor, data_test)
  plot(data_test[,1],type='l',col='blue')
  lines(ts(y_pred),type='l',col='red')
  if (i==1) forecastdata<-data.frame(y_pred)
  else forecastdata[,i]<-y_pred
}


write.xlsx(forecastdata,file='C:/Users/不是杨花/Desktop/forecast19.xlsx')

#################################################SVM


obj <- tune(svm, Y~., data = data_trn, 
            ranges = list(gamma = 2^(-1:1), cost = 2^(2:4)),
            tunecontrol = tune.control(sampling = "fix")
)

obj <- tune.svm(Y~., data = data_trn, gamma = 2^(-1:1), cost = 2^(2:4))
summary(obj)
plot(obj)

y_pred = predict(obj, data_test)
plot(data_test[,1],type='l',col='blue')
lines(ts(y_pred),type='l',col='red')
















#################################################EEMD


# 设置工作空间
# 把“数据及程序”文件夹拷贝到 F 盘下，再用 setwd 设置工作空间
setwd("D:/桌面文件/中南财经政法/期刊论文/基于ICEEMDAN-CWOA-LSTM算法的集成学习方法预测大宗商品期货价格")
library(openxlsx)

data1 <- read.xlsx('期货收盘价NYMEX轻质原油.xlsx')
x1<-ts(data1$price)
#' @importFrom Rlibeemd emd_num_imfs eemd
#' @importFrom forecast nnetar forecast
#' @importFrom utils head tail
#' @importFrom graphics plot
#' @importFrom stats as.ts ts
#' @export
library(Rlibeemd)
library(forecast)
library(utils)




EEMD <- function(data, stepahead=20, num.IMFs=10,
                 s.num=4L, num.sift=50L, ensem.size=250L, noise.st=0.2) {
  n.IMF <- num.IMFs
  AllIMF <- eemd(ts(data), num_imfs = n.IMF, ensemble_size = ensem.size, noise_strength = noise.st,
                 S_number = s.num, num_siftings = num.sift, rng_seed = 0L, threads = 0L)
  
  
  Plot_IMFs <- AllIMF
  AllIMF_plots <- plot(Plot_IMFs)
  return(list(TotalIMF = n.IMF, AllIMF=AllIMF,AllIMF_plots=AllIMF_plots))
}

# 鑾峰彇IMF
x_9IMF<-EEMD(x1)   # x_3IMF<-EEMDTDNN(x1)  data_3IMF<-x_3IMF$AllIMF  write.xlsx(data_3IMF,file='C:/Users/涓嶆槸鏉ㄨ姳/Desktop/缁熻寤烘ā/鏁版嵁闆?/data_3IMF.xlsx')
data_9IMF<-x_9IMF$AllIMF
write.xlsx(data_9IMF,file='D:/桌面文件/中南财经政法/期刊论文/基于ICEEMDAN-CWOA-LSTM算法的集成学习方法预测大宗商品期货价格/EEMD分解结果.xlsx') 




#################################################################################检验序列平稳和白噪声性

# 设置工作路径
setwd("D:/桌面文件/中南财经政法/期刊论文/基于分解-重构策略的神经网络预测大宗商品期货价格方法")
library(openxlsx)

## T1
data <- read.xlsx('IMFs-prediction-results.xlsx')

x <- ts(data$`pre-Res`)
plot(x,main = '时序图')
library(aTSA)
adf.test(x)     #平稳性检验
for(k in 1:2)print(Box.test(x,lag=6*k))      #白噪声检验














#################################################################################ICEEMDAN分解后的聚类


# 设置工作空间
# 把“数据及程序”文件夹拷贝到F盘下，再用setwd设置工作空间
setwd("D:/桌面文件/中南财经政法/期刊论文/基于分解-重构策略的神经网络预测大宗商品期货价格方法")
library(openxlsx)
library(TSA)
data_9 <- read.xlsx('ICEEMDAN分解结果.xlsx',sheet=1)

# 绘制相关系数矩阵图
library(corrplot)
######1-4
m<-as.matrix(data_9[,3:9])   # rownames(m)<-data2[,1]
r<-cor(m)

corrplot(r,order='AOE',addCoef.col='grey20')             
corrplot(r,type='upper',method='pie',order='AOE',tl.pos='d',cl.pos='b') # 上半角绘制圆
corrplot(r,add=TRUE,type='lower',method='number',order='AOE',diag=FALSE,tl.pos='n',cl.pos='n')  # 下半角绘制相关系数
library(PerformanceAnalytics)
chart.Correlation(data_9[,1:9], histogram=TRUE, pch=19)

# 聚类分析
#######9-6
cluster_month <- dist(scale(r[1:7]),method="euclidean")
hcluster_month <- hclust(cluster_month,method="ward.D2")
par(mai=c(0.8,0.6,0.2,0),cex=0.9)
######'IMF1','IMF2',
plot(hcluster_month,cex=0.7,labels=c('IMF3','IMF4','IMF5','IMF6','IMF7','IMF8','IMF9'),hang=-0.1)
rect.hclust(hcluster_month,k=3)







############################################################################ DM检验
setwd("D:/桌面文件/中南财经政法/期刊论文/基于ICEEMDAN-CWOA-LSTM算法的集成学习方法预测大宗商品期货价格")
library(openxlsx)
dataset <- read.xlsx('DM检验.xlsx',sheet=2)       #1是一步 2是三步 3是六步
P=matrix(0, nrow = 9, ncol = 1)
DM=matrix(0,9,1)

for (i in 1:9) {
  
  #d=dataset[,1]-dataset[,i]
  d=(dataset[,10]-dataset[,8])**2-(dataset[,10]-dataset[,i])**2
  d_mean=abs(mean(d))
  d_std=((239)**(-1/2))*sd(d-d_mean)  #251 239
  DM[i]=d_mean/d_std
  P[i]=1-pnorm(DM[i])
}
DM
P











































































# 根据聚类分析结果，对 IMF 进行重构，分为高中低频三项
## 此项工作在 excel 中完成，与直接输出没有太大差异，但由于如果只选择输出三项，
则 residual 项没法很好的剥离趋势
#################################################################绘图小工作
# 绘制合并时序图
# 把“数据及程序”文件夹拷贝到 F 盘下，再用 setwd 设置工作空间
setwd("C:/Users/不是杨花/Desktop/统计建模/数据集 2")
library(openxlsx)
data2 <- read.xlsx('datazhen.xlsx')
########## 绘制大时序图
x1=ts(data2$IMF1,frequency = 48,start = c(2010,24),end = c(2021,22))
x2=ts(data2$IMF2,frequency = 48,start = c(2010,24),end = c(2021,22))
x3=ts(data2$IMF3,frequency = 48,start = c(2010,24),end = c(2021,22))
r=ts(data2$Residual,frequency = 48,start = c(2010,24),end = c(2021,22))
total=ts(data2$total,frequency = 48,start = c(2010,24),end = c(2021,22))
par(mai=c(0.8,0.8,0.4,0.4))
ts.plot(x1,x2,x3,r,total,gpars = list(xlab='年份',ylab='WTI 原油周价格'))





































# 设置工作空间
# 把“数据及程序”文件夹拷贝到F盘下，再用setwd设置工作空间
setwd("D:/桌面文件/中南财经政法/期刊论文/基于ICEEMDAN-CWOA-LSTM算法的集成学习方法预测大宗商品期货价格")
library(openxlsx)
data <- read.xlsx('六步输出.xlsx',sheet=11)
x1<-ts(head((data[,1]),5766)) 




stepahead=251
data_trn <- ts(head(data, round(length(data[,1]) - stepahead))) # 修正length[data]为length(data[,1])
data_test <- ts(tail(data, stepahead))
IMF_trn <- data[-c(((length(data[,1])-stepahead)+1):length(data[,1])),] # 修正length[data]为length(data[,1])


library(forecast)

# 分量一预测效果的评估
fit1<-auto.arima(ts(x1))
# tsdiag(fit1)
fore1<-forecast::forecast(fit1,h=251)
MAE_arima1=mean(abs(data_test[,1] - as.numeric(fore1$mean)))
MAPE_arima1=mean(abs(  (data_test[,1] - as.numeric(fore1$mean))/data_test[,1]))
rmse_arima1=sqrt(mean((data_test[,1] - as.numeric(fore1$mean))^2))


ts.plot(as.numeric(fore1$mean),data_test[,1],gpars=list(col=c("blue","red")))












# 加载所需要的包
library(forecast)

# 读取时间序列数据
#data <- read.csv("data.csv")
#ts_data <- ts(data, start=c(2010,1), frequency=12)  # 将数据转换为时间序列格式
ts_data=x1

# 拟合ARIMA模型，并检验其残差是否符合正态分布
fit <- auto.arima(ts_data)  # 拟合ARIMA模型
Box.test(fit$residuals, lag=log(length(fit$residuals)))  # 检验其残差是否符合正态分布，p-value>0.05表明符合正态分布。

# 预测251期的时间序列数据，并评估预测效果。 
pred <- forecast(fit, h=251)  # 预测251期的时间序列数据。 
accuracy(pred,data_test[,1])  # 评估预测效果

plot(ts_data, type="l", col="red", xlab="Time", ylab="Value") # 画出第一个时间序列x，并用红色标识

lines(ts_data, type="l", col="blue") # 画出第二个时间序列y，并用蓝色标识




























######################################ELM


# 设置工作空间
# 把“数据及程序”文件夹拷贝到F盘下，再用setwd设置工作空间
setwd("D:/桌面文件/中南财经政法/期刊论文/基于ICEEMDAN-CWOA-LSTM算法的集成学习方法预测大宗商品期货价格")
library(openxlsx)
data <- read.xlsx('未分解多步输出.xlsx',sheet=1)
x1<-ts(head((data[,1]),5772)) 




stepahead=251
data_trn <- ts(head(data, round(length(data[,1]) - stepahead))) # 修正length[data]为length(data[,1])
data_test <- ts(tail(data, stepahead))
IMF_trn <- data[-c(((length(data[,1])-stepahead)+1):length(data[,1])),] # 修正length[data]为length(data[,1])


library(forecast)

IndIMF<-IMF_trn

EEMDELMFit <- nnfor::elm(as.ts(data_test ), keep = NULL, difforder = NULL, outplot = c(FALSE), sel.lag = c(FALSE), direct = c(FALSE), allow.det.season = c(FALSE))
EEMDELM_fcast=forecast::forecast(EEMDELMFit, h=stepahead)
EEMDELM_fcast_Mean=EEMDELM_fcast$mean
Fcast_AllIMF <- EEMDELM_fcast_Mean

MAE_EEMDelm1=mean(abs(ts(data_test[,1]) - ts(Fcast_AllIMF)))
MAPE_EEMDelm1=mean(abs( ts(data_test[,1]) - ts(Fcast_AllIMF))/ts(data_test[,1]))
rmse_EEMDelm1=sqrt(mean(ts(data_test[,1]) - ts(Fcast_AllIMF))^2)


plot(ts(Fcast_AllIMF), type="l", col="red", xlab="Time", ylab="Value") # 画出第一个时间序列x，并用红色标识

lines(ts(data_test[,1]), type="l", col="blue") # 画出第二个时间序列y，并用蓝色标识








