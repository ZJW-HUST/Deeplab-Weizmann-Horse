# DeeplabV3+
使用DeeplabV3+在Weizmann Horse数据集上训练对马的二分类语义分割网络
<p align="center">
  <img src=".\weizmann_horse_db\horse_test\horse282.png" width="250" title="Original Image"/>
  <img src=".\result\m282.png" width="250" title="gt"/>
  <img src=".\result\282.png" width="250" title="Prediction"/>
</p>
<p align="center">
  <img src=".\weizmann_horse_db\horse_test\horse285.png" width="260" title="Original Image"/>
  <img src=".\result\m285.png" width="260" title="gt"/>
  <img src=".\result\285.png" width="260" title="Prediction"/>
</p>

# 模型各文件说明
运行可视化.ipynb文件即可查看模型输出可视化结果  
train.py为主文件，包含了Trainer类，里面实例化了DeepLab对象，调用trainer.training(epoch)即可完成一次epoch的训练，调用trainer.validation(epoch)可完成对训练集的测试，并得到各类指标。  
modeling包含了DeepLab类以及其使用的各种模型  
mydataloaders用于加载和生成数据集  
utils包含了测试指标的计算、lr_scheduler、loss函数的定义、各类训练权重的计算、模型的保存、损失的可视化等内容   
weizmann_horse_db包含了训练集和验证集   
若运行train.py文件对模型进行训练，会先下载预训练的backbone，最后创建一个run文件夹，将模型最优参数、训练过程中相关信息(events.out.tfevents文件)保存
 # 环境依赖
 需要 python3.7 pytorch1.1.0 pillow cv2 
 # 使用训练好的模型测试
 百度网盘地址：
 链接：https://pan.baidu.com/s/1mkuX0c3YaIuAFb2Bohcndg?pwd=84ov 
 提取码：84ov
# 训练过程截图
<img src=".\result\res.png" title="训练过程截图"/>
