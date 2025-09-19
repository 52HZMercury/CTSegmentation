#### 项目描述
用于CT影像分割的模型，目前集成了basicunet,segmamba

#### 部署项目
1. 创建虚拟环境
```angular2html
conda create -n [your_env_name] python=3.8
```
2. 安装环境依赖包
```angular2html
pip install -r requirements.txt
```
3. 使用read_dir.py生成数据目录json文件
```angular2html
python utils/read_dir.py
```
4. 启动训练
```angular2html
python main.py
```