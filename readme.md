# 移动物联网应用系统开发课程设计

# 期末项目 预测模型部分

### 创立以下文件夹：

---- code					//代码文件

---- datafeature			//生成的特征文件

---- dataset				//初始数据集

---- datasmoted			//经过均衡化处理后的特征文件

---- model				//生成的模型文件

---- result			 	//预测的结果

#### 请在dataset文件夹内放置以下文件

- 训练集1 traindata_N15_M01_F10.csv
- 训练集2 traindata_N15_M07_F04.csv
- 测试集B testdataB.csv

### 运行流程

- 在code文件夹下打开feature_get.py并运行
  - 获取训练集traindata_N15_M01_F10.csv、traindata_N15_M07_F04.csv的特征
  - 将获得的部分特征（12个）输出到feature文件夹中的csv文件
- 在code文件夹下打开data_smote.py并运行
  - 将上一部获得的traindata_N15_M01_F10.csv的特征文件进行数据均衡化处理，便于后续的训练
- 在code文件夹下打开testdata_feature_get.py并运行
  - 获取测试集testdataA、testdataB的特征并输出
- 在code文件夹下打开Random_Forest.py并运行
  - 采用随机森林算法生成模型
  - 将第二个训练集作为测试，获得模型的评分
- 在code文件夹下打开result_prediction.py并运行
  - 利用上一步生成的模型，对测试集进行预测并输出结果

### Othercode

- 其他的代码主要是在模型建立前期，提取所有45个特征后进行筛选的过程，同时绘制了特征贡献度图
- othercode中的路径均为绝对路径，可能无法运行 
