# LoRA 微调Llama2

本项目使用LoRA对Llama2-7B进行微调，使用了FinancialPhraseBank数据集。该过程包括以下步骤：

1. **数据预处理**：将原始文本文件转换为结构化的CSV格式数据，用于模型训练。
2. **模型微调**：使用LoRA对模型进行微调，以适应特定任务。
3. **评估和测试**：使用微调后的模型进行预测并评估其性能。

## 项目结构

```
project/
│
├── datasets/                 
|   ├── raw_datas/             # 存放原始文本文件的目录，用于数据预处理
|   |   ├── Sentences_50Agree.txt
|   |   ├── Sentences_66Agree.txt
|   |   ├── Sentences_75Agree.txt
|   |   └── Sentences_AllAgree.txt
|   ├── all-data.csv          # 预处理后的CSV文件，包含所有数据
│   └── preprocess_data.py    # 数据预处理脚本，将原始数据转换为CSV格式
│
├── utils/                     # 用于数据加载、评估、预测等功能的工具类
│   ├── get_data.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── get_prompt.py
│   └── args.py
│      
├── train_lora.py              # 使用LoRA对模型进行微调的脚本
└── test_lora.py               # 测试并评估微调后的模型的脚本
```

## 环境要求

- Python 3.11
- PyTorch 2.1.2
- CUDA 12.2
- Ubuntu 22.04
- 其他依赖：
  - `pandas==2.2.3`
  - `chardet==5.2.0`
  - `trl==0.14.0`
  - `transformers==4.48.3`
  - `datasets==3.2.0`
  - `peft==0.14.0`
  - `scikit-learn==1.6.1`
  - `bitsandbytes==0.45.2`
  - `numpy==1.26.3`

你可以使用以下命令安装所需的其他依赖库：

```bash
pip install -r requirements.txt
```

## 使用方法

### 第一步：数据预处理

`preprocess_data.py` 脚本将原始文本文件转换为结构化的CSV文件。原始文本文件应存放在 `raw_datas/` 目录下，每行数据格式应为 `sentence @sentiment`。

运行数据预处理脚本：

```bash
cd dataset
python preprocess_data.py
cd ..
```

该脚本将生成一个 `all_data.csv` 文件，供训练时使用。或者直接使用 `all-data.csv` 文件进行训练。

### 第二步：微调模型（LoRA）

需要从 [Hugging Face Model Hub](https://huggingface.co/models) 下载预训练模型。该脚本使用的是`llama-2-7b-hf`模型，请自行下载并放置在 `./model/` 目录下。


完成数据预处理后，可以开始微调预训练语言模型。

1. 修改 `args.py` 配置文件中的训练参数。
2. 运行微调脚本：

```bash
python train_lora.py --model_name model
```

该脚本将加载预训练模型，并使用LoRA技术进行高效微调，微调后的模型将保存到指定的输出目录。

### 第三步：评估微调后的模型

微调完成后，你可以使用 `test_lora.py` 脚本评估模型的性能。

1. 修改 `args.py` 配置文件中的评估参数。
2. 运行测试脚本：

```bash
python test_lora.py --merged_model_dir <微调后模型的路径>
```

该脚本将使用微调后的模型进行预测，并评估其性能。
