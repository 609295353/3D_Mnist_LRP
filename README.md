# 3D_Mnist_LRP
使用3DCNN对3D的mnist数据集进行分类，将模型的分类决策通过LRP分解到原图像的每一个像素点上并以heatmap（热力图）加以展示

# 数据描述
此次实验使用的是3D-Mnist数据集，总共包含12000个样本数据，类别数量为10，数据样本分布如下表所示，每个样本为16*16*16的三维图像数组。
类别	数量

0-1128

1-1378

2-1208

3-1200

4-1290

5-1042

6-1176

7-1298

8-1102

9-1178


# model.py 模型结构
结构如下

input

conv-32

conv-64

max-pooling

conv-128

conv-256

max-pooling

dropout-0.5

fc-4096

dropout-0.75

fc-1024

dropout-0.75

fc-10

softmax

output

# 2dresult
2d的切片heatmap显示

# 3dresult
3d的heatmap三维空间显示

# 结果
训练的模型最终在测试集上精度为0.8左右，训练过程记录图如train_test_result.png所示，2dresult,3dresult文件夹中放的是经过lrp之后得到的heatmap

# 使用方法
运行train.py后得到训练模型，再运行lrp.py即可，结果训练过程记录图保存为train_test_result.png，heatmap存储在2dresult、3dresult文件夹中
