# EmbeddingModelDistillation

使用开源中文数据集蒸馏现有嵌入模型，来提高模型效率

本项目实现了基于知识蒸馏的嵌入模型压缩技术，使用 [CSTS (Chinese STS-B) 数据集](https://github.com/zejunwang1/CSTS) 作为蒸馏数据集，参考 [llm_related/knowledge_distillation_embedding](https://github.com/wyf3/llm_related/tree/main/knowledge_distillation_embedding) 中的方法进行模型训练。

## 特性

- 🎯 **知识蒸馏**: 使用大型教师模型（qzhou-embedding）指导小型学生模型（qwen3-embedding-4b）学习
- 📊 **中文数据集**: 基于 CSTS 中文语义相似度数据集进行训练
- 🚀 **高效训练**: 支持梯度累积、混合精度训练等优化技术
- 📈 **全面评估**: 提供多种评估指标（Spearman、Pearson、准确度等）
- 🔧 **灵活配置**: 支持YAML配置文件和命令行参数
- 📝 **详细日志**: 集成 loguru 和 Weights & Biases 日志记录

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/MiningIrving/EmbeddingModelDistillation.git
cd EmbeddingModelDistillation
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 安装包（可选）：
```bash
pip install -e .
```

## 快速开始

### 1. 准备数据

使用示例数据：
```bash
python src/main.py --mode prepare_data --create_sample_data
```

或下载真实的 CSTS 数据集：
```bash
python src/main.py --mode prepare_data --download_data
```

### 2. 训练模型

使用默认配置训练：
```bash
python src/main.py --mode train --create_sample_data
```

使用自定义参数：
```bash
python src/main.py \
    --mode train \
    --config configs/default_config.yaml \
    --teacher_model "Qwen/Qwen2.5-7B-Instruct" \
    --student_model "Qwen/Qwen2.5-3B-Instruct" \
    --num_epochs 5 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --output_dir ./output
```

### 3. 评估模型

```bash
python src/main.py --mode evaluate --create_sample_data
```

## 配置文件

配置文件示例 (`configs/default_config.yaml`):

```yaml
# 数据集配置
dataset:
  name: "csts"
  data_dir: "./data/csts"
  max_length: 512
  batch_size: 32

# 模型配置
models:
  teacher:
    name: "qzhou-embedding"
    model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
  student:
    name: "qwen3-embedding-4b"
    model_name_or_path: "Qwen/Qwen2.5-3B-Instruct"

# 训练配置
training:
  output_dir: "./output"
  num_epochs: 5
  learning_rate: 2e-5
  weight_decay: 0.01

# 蒸馏配置
distillation:
  temperature: 4.0
  alpha: 0.7  # 蒸馏损失权重
  beta: 0.3   # 真实标签损失权重
  loss_type: "mse"
```

## 项目结构

```
EmbeddingModelDistillation/
├── src/
│   ├── data/                   # 数据加载和预处理
│   │   ├── csts_dataset.py    # CSTS数据集实现
│   │   └── data_utils.py      # 数据工具函数
│   ├── models/                # 模型实现
│   │   ├── embedding_model.py # 基础嵌入模型
│   │   ├── teacher_model.py   # 教师模型
│   │   └── student_model.py   # 学生模型
│   ├── training/              # 训练相关
│   │   ├── distillation_trainer.py  # 蒸馏训练器
│   │   ├── losses.py          # 损失函数
│   │   └── utils.py           # 训练工具
│   ├── evaluation/            # 评估模块
│   │   ├── metrics.py         # 评估指标
│   │   └── evaluator.py       # 模型评估器
│   ├── utils/                 # 通用工具
│   └── main.py               # 主入口点
├── configs/                   # 配置文件
├── examples/                  # 示例脚本
├── tests/                     # 测试文件
└── requirements.txt          # 依赖列表
```

## 使用示例

### 训练示例

```python
from src.data import create_csts_datasets, create_sample_data
from src.models import TeacherModel, StudentModel
from src.training import DistillationTrainer
from src.utils import load_config

# 加载配置
config = load_config("configs/default_config.yaml")

# 准备数据
create_sample_data("./data/csts")

# 加载模型
teacher_model = TeacherModel("Qwen/Qwen2.5-7B-Instruct")
student_model = StudentModel("Qwen/Qwen2.5-3B-Instruct")

# 创建数据集
datasets = create_csts_datasets(
    data_dir="./data/csts",
    tokenizer=student_model.tokenizer
)

# 训练
trainer = DistillationTrainer(
    teacher_model=teacher_model,
    student_model=student_model,
    train_dataset=datasets["train"],
    eval_dataset=datasets.get("dev"),
    config=config
)

results = trainer.train()
```

### 评估示例

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_distillation(
    teacher_model=teacher_model,
    student_model=student_model,
    dataset=datasets["test"]
)

print(f"Teacher Spearman: {results['teacher_results']['similarity_metrics']['spearman']:.4f}")
print(f"Student Spearman: {results['student_results']['similarity_metrics']['spearman']:.4f}")
```

## 支持的模型

### 教师模型 (qzhou-embedding)
- 基于 Qwen2.5-7B-Instruct 实现
- 提供高质量的嵌入表示作为蒸馏目标

### 学生模型 (qwen3-embedding-4b)  
- 基于 Qwen2.5-3B-Instruct 实现
- 更小的模型规模，推理速度更快
- 通过蒸馏学习教师模型的知识

## 评估指标

- **Spearman 相关系数**: 排序相关性
- **Pearson 相关系数**: 线性相关性  
- **准确度**: 二分类准确度
- **F1 分数**: 综合评估指标
- **MSE/MAE**: 回归误差指标

## 日志和监控

### 本地日志
- 控制台输出彩色日志
- 文件日志自动轮转和清理
- 支持不同日志级别

### Weights & Biases 集成
```bash
python src/main.py --mode train --wandb
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- [CSTS 数据集](https://github.com/zejunwang1/CSTS) 提供中文语义相似度数据
- [llm_related](https://github.com/wyf3/llm_related) 项目的知识蒸馏方法参考
- Qwen 系列模型提供强大的基础模型支持
