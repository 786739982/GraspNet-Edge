#：与官方的安装流程有所出入，因为在官方的基础上修改了一些编译的bug，并添加了一些文件

# 安装步骤

1、
    安装对应的版本的pytorch 2.1 版本和torchvision

2、
    cd graspnet-baseline
    pip install -r requirements.txt

3、
    cd pointnet2
    python setup.py install

4、
    cd knn
    python setup.py install

5、
    cd graspnetAPI
    pip install .

# 测试是否环境安装成功
6、
    bash command_demo.sh 

# 已有材料说明：
7、
    自己训练的 checkpoint 文件 在 logs/log_dist_lr0.005_bs16
    之前写的 onnx 导出程序是 to_onnx.py
    目前的该版本的是没有替换C++算子的graspnet版本，所以运行to_onnx.py后导出的grasp.onnx可视化的时候输入没有接入网络。需要替换算子为python版本的

# 数据集来源
    Graspnet 官方数据集  https://graspnet.net/