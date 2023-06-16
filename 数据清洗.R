
#一、数据清洗
setwd("D:/桌面文件/中南财经政法/期刊论文/基于分解-重构策略的神经网络预测大宗商品期货价格方法") # 设置工作路径
library(openxlsx)
data1 <- read.xlsx('WTI期货价格及影响因素原始数据.xlsx')

data_aqi <- data1[which(data1$type == 'AQI'),]   


# 2. 处理缺失值                                          
# 2.1检测数据是否有缺失值  
library(mice)                                           
md.pattern(data1)                              #生成一个展示缺失值模式的表格
# 2.2可视化识别缺失值
library(VIM)
aggr(data1,prop = F , numbers = T)               ####绝对结果显示植物园有大量缺失值
aggr(data1,prop = T , numbers = T)               ####相对结果显示植物园有大量缺失值
matrixplot(data1)                              #### 更清晰直观的看到总体缺失情况

data1=na.omit(data1)

write.xlsx(data1,file='D:/桌面文件/中南财经政法/期刊论文/基于分解-重构策略的神经网络预测大宗商品期货价格方法/Original_data.xlsx')


























