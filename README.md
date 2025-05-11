# BiGRU-Multihead-Attention 分类模型

这是一个基于 PyTorch 实现的双向 GRU + 多头注意力机制的分类模型，翻译自 MATLAB 版本。模型可以在 CPU 环境下运行。

## 项目结构

- `main.py` - 主程序，负责数据加载、模型训练和评估
- `model.py` - 模型定义，包含 BiGRU-Attention 架构
- `flip_layer.py` - 自定义翻转层，用于双向 GRU 实现
- `polygon_area_metric.py` - 模型评价指标计算
- `setup_env.sh` - 环境配置脚本（macOS/Linux）

## 快速环境配置

项目提供了自动配置环境的脚本，只需一个命令即可完成所有环境设置：

```bash
# 添加执行权限
chmod +x setup_env.sh

# 执行脚本
./setup_env.sh
```

脚本将自动执行以下操作：
1. 检查并删除已存在的环境（如果有）
2. 创建新的 Python 3.9 环境
3. 安装所有必要的依赖
4. 验证安装是否成功


## 使用方法

1. 确保数据文件 `System.xlsx` 位于当前目录
2. 运行主程序：

```bash
python main.py
```

程序将执行以下步骤：
1. 加载数据并按类别分割成训练集和测试集
2. 构建和训练 BiGRU-Multihead-Attention 模型
3. 在训练集和测试集上评估模型性能
4. 显示各种性能指标和可视化结果

## 数据格式

输入数据应为 Excel 表格，其中：
- 每行代表一个样本
- 最后一列是样本的类别标签
- 前面的列是特征数据

## 模型架构

该模型包含以下主要组件：
- 前向 GRU 层
- 反向 GRU 层（通过翻转输入序列实现）
- 多头自注意力机制
- 全连接层

## 性能评估

模型评估使用多种指标：
- 多边形面积 (PAM)
- 分类准确率
- 灵敏度
- 特异性
- ROC 曲线下面积 (AUC)
- Kappa 系数
- F-measure 