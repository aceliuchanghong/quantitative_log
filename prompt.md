提示词

---

```
以老师教导学生的风格，教授我，循序渐进，言简意赅。  
1. 逐步引导，推导核心概念
2. 深入讲解关键细节或进阶内容，保持清晰。[非常重要--深入讲解]  
3. 最后提供相关代码，并简要解释代码逻辑。  
确保每步逻辑连贯，语言亲切，像老师一样耐心引导。
```

```
给出对应模块的python代码, 需要逻辑连贯、层层递进, 系统拆分为高内聚、低耦合的模块
```

---

我在想构建一个**滚动递推的半日级极值预测系统**，基于股票历史分钟级数据，每日分上午（9:30–11:30, $\text{am}$）与下午（13:00–15:00, $\text{pm}$）两个时段，聚合为特征向量：

$$
\mathbf{x}_t^{(s)} = \left[ \text{OHLCV}, \text{RSI}, \text{MACD}, \sigma_{\text{realized}}, \dots \right]^\top \in \mathbb{R}^{d_0}
$$

并**增强输入特征**为：

$$
\mathbf{\tilde{x}}_t^{(s)} = \left[ \mathbf{x}_t^{(s)}; \mathbf{z}_t^{(s)}; \boldsymbol{\tau}_t^{(s)} \right] \in \mathbb{R}^d, \quad d = d_0 + d_0 + d_\tau
$$

其中：

1. **原始行情与技术特征**：$\mathbf{x}_t^{(s)} \in \mathbb{R}^{d_0}$  
   → 例如 OHLCV、RSI、MACD、波动率等，维度为 $d_0$

2. **标准化后的 z-score 特征**：$\mathbf{z}_t^{(s)} \in \mathbb{R}^{d_0}$  
   → 和原始特征一一对应，是对 $\mathbf{x}_t^{(s)}$ 的标准化版本，维度也为 $d_0$

3. **时间编码特征**：$\boldsymbol{\tau}_t^{(s)} \in \mathbb{R}^{d_\tau}$  
   → 如星期几、是否月末、节假日标记等，维度为 $d_\tau$

所以拼接后的总维度是：

> 原始特征维度 + 标准化特征维度 + 时间编码维度 = $d_0 + d_0 + d_\tau$

- $\mathbf{z}_t^{(s)} = \dfrac{ \mathbf{x}_t^{(s)} - \boldsymbol{\mu}_t^{(s)} }{ \boldsymbol{\sigma}_t^{(s)} }$ 为**20日滚动标准化特征**（$\boldsymbol{\mu}_t^{(s)}, \boldsymbol{\sigma}_t^{(s)}$ 基于 $t-20$ 至 $t-1$ 计算）；
- $\boldsymbol{\tau}_t^{(s)} \in \mathbb{R}^{d_\tau}$ 为**结构化时间编码**，包含：
  - 星期几（one-hot），
  - 是否月末前3日（布尔），
  - 是否节假日前/后1日（布尔），
  - 时段标识（am/pm, 1维），
  - 交易日序号模周期（如月内第几个交易日）等；

同时引入外部协变量 $\mathbf{c}_t^{(s)} \in \mathbb{R}^{d_c}$（如大盘指数收益率、行业动量、关联个股特征等），最终模型输入为：

$$
\mathbf{u}_t^{(s)} = \left[ \mathbf{\tilde{x}}_t^{(s)}; \mathbf{c}_t^{(s)} \right] \in \mathbb{R}^{d + d_c}
$$

**预测流程采用两阶段滚动机制：**

1. **上午预测（$t$ 日 am）**：输入最近5个交易日共10个半日样本：
   $$
   \mathcal{U}_{t-5:t-1} = \left\{ \mathbf{u}_{t-5}^{\text{am}}, \mathbf{u}_{t-5}^{\text{pm}}, \dots, \mathbf{u}_{t-1}^{\text{am}}, \mathbf{u}_{t-1}^{\text{pm}} \right\}
   $$
   预测目标：
   $$
   \hat{y}_t^{\text{am}} = \left( \hat{H}_t^{\text{am}}, \hat{L}_t^{\text{am}} \right) = f_\theta \left( \mathcal{U}_{t-5:t-1} \right)
   $$

2. **下午预测（$t$ 日 pm）**：将上午预测值或真实观测值 $\mathbf{u}_t^{\text{am}}$ 加入窗口，形成“4.5天+0.5天”序列：
   $$
   \mathcal{U}_{t-4.5:t} = \left\{ \mathbf{u}_{t-5}^{\text{pm}}, \mathbf{u}_{t-4}^{\text{am}}, \dots, \mathbf{u}_{t-1}^{\text{pm}}, \mathbf{u}_{t}^{\text{am}} \right\}
   $$
   预测目标：
   $$
   \hat{y}_t^{\text{pm}} = \left( \hat{H}_t^{\text{pm}}, \hat{L}_t^{\text{pm}} \right) = f_\theta \left( \mathcal{U}_{t-4.5:t} \right)
   $$

3. **滚动更新机制**：完成 $t$ 日预测后：
   - 若使用真实数据，将 $\mathbf{u}_t^{\text{pm}}$ 加入历史；
   - 同步更新 $\boldsymbol{\mu}_{t+1}^{(s)}, \boldsymbol{\sigma}_{t+1}^{(s)}$（滑动窗口移除 $t-20$，加入 $t$）；
   - 更新 $\boldsymbol{\tau}_{t+1}^{(s)}$ 依据日历与交易日历；
   - 窗口前移：
     $$
     \mathcal{U}_{t-4:t} \leftarrow \left( \mathcal{U}_{t-4.5:t} \setminus \{ \mathbf{u}_{t-5}^{\text{am}} \} \right) \cup \{ \mathbf{u}_{t}^{\text{pm}} \}, \quad t \leftarrow t + 1
     $$

实现**时序感知 + 统计归一化 + 时间结构编码 + 市场协同驱动 + 滚动自更新**的完整闭环预测系统，用于金融时序的极值滚动预测任务。


---

为实现对股票日内价格极值（高点/低点）的精准滚动预测，我们设计一个**时序感知 + 统计归一化 + 时间结构编码 + 市场协同驱动 + 滚动自更新**的闭环预测系统。该系统基于历史分钟级数据，以半日（上午 9:30–11:30，下午 13:00–15:00）为基本预测单元，构建滚动递推预测机制。

首先，对每个半日时段 $ s \in \{ \text{am}, \text{pm} \} $，基于分钟级数据聚合形成原始特征向量：

$$
\mathbf{x}_t^{(s)} = \left[ \text{OHLCV}, \text{RSI}, \text{MACD}, \sigma_{\text{realized}}, \dots \right]^\top \in \mathbb{R}^{d_0}
$$

其中包含价格、成交量、技术指标与波动率等，维度为 $ d_0 $。

为进一步提升模型泛化能力，我们**增强输入特征**，构造：

$$
\mathbf{\tilde{x}}_t^{(s)} = \left[ \mathbf{x}_t^{(s)}; \mathbf{z}_t^{(s)}; \boldsymbol{\tau}_t^{(s)} \right] \in \mathbb{R}^d, \quad d = d_0 + d_0 + d_\tau
$$

该增强向量由三部分拼接而成：

1. **原始行情与技术特征**：$\mathbf{x}_t^{(s)} \in \mathbb{R}^{d_0}$  
   → 包含 OHLCV、RSI、MACD、已实现波动率等，维度为 $ d_0 $

2. **标准化后的 z-score 特征**：$\mathbf{z}_t^{(s)} \in \mathbb{R}^{d_0}$  
   → 与原始特征一一对应，为滚动标准化版本：
   $$
   \mathbf{z}_t^{(s)} = \dfrac{ \mathbf{x}_t^{(s)} - \boldsymbol{\mu}_t^{(s)} }{ \boldsymbol{\sigma}_t^{(s)} }
   $$
   其中均值与标准差基于最近20个交易日（$t-20$ 至 $t-1$）滚动计算，确保统计稳定性。

3. **时间编码特征**：$\boldsymbol{\tau}_t^{(s)} \in \mathbb{R}^{d_\tau}$  
   → 引入结构性时间信息，包含：
   - 星期几（one-hot 编码），
   - 是否月末前3日（布尔标记），
   - 是否节假日前/后1日（布尔标记），
   - 时段标识（am/pm，1维），
   - 月内交易日序号（模周期编码）等。

因此，增强特征总维度为：
> $ d = d_0 + d_0 + d_\tau $

为捕捉市场整体动量与行业联动效应，我们进一步引入外部协变量：

$$
\mathbf{c}_t^{(s)} \in \mathbb{R}^{d_c}
$$

例如：大盘指数收益率、行业动量因子、关联个股特征、资金流指标等。

最终，模型输入向量为：

$$
\mathbf{u}_t^{(s)} = \left[ \mathbf{\tilde{x}}_t^{(s)}; \mathbf{c}_t^{(s)} \right] \in \mathbb{R}^{d + d_c}
$$

预测采用**上午 → 下午 → 滚动更新**的两阶段递推结构，确保信息流与时序一致性。

输入：最近5个交易日（共10个半日）的历史特征序列：

$$
\mathcal{U}_{t-5:t-1} = \left\{ \mathbf{u}_{t-5}^{\text{am}}, \mathbf{u}_{t-5}^{\text{pm}}, \dots, \mathbf{u}_{t-1}^{\text{am}}, \mathbf{u}_{t-1}^{\text{pm}} \right\}
$$

输出：预测上午时段极值：

$$
\hat{y}_t^{\text{am}} = \left( \hat{H}_t^{\text{am}}, \hat{L}_t^{\text{am}} \right) = f_\theta \left( \mathcal{U}_{t-5:t-1} \right)
$$

在上午预测完成后，将上午时段的真实观测值（或预测值，视策略而定）$\mathbf{u}_t^{\text{am}}$ 加入输入窗口，形成“4.5天 + 0.5天”的滚动序列：

$$
\mathcal{U}_{t-4.5:t} = \left\{ \mathbf{u}_{t-5}^{\text{pm}}, \mathbf{u}_{t-4}^{\text{am}}, \dots, \mathbf{u}_{t-1}^{\text{pm}}, \mathbf{u}_{t}^{\text{am}} \right\}
$$

输出：预测下午时段极值：

$$
\hat{y}_t^{\text{pm}} = \left( \hat{H}_t^{\text{pm}}, \hat{L}_t^{\text{pm}} \right) = f_\theta \left( \mathcal{U}_{t-4.5:t} \right)
$$

完成 $t$ 日预测后，系统执行以下更新操作，确保下一交易日预测的连续性与适应性：

- **数据更新**：将 $t$ 日下午的真实观测 $\mathbf{u}_t^{\text{pm}}$ 加入历史序列；
- **统计参数更新**：滑动窗口移除 $t-20$ 日数据，加入 $t$ 日数据，重新计算 $\boldsymbol{\mu}_{t+1}^{(s)}, \boldsymbol{\sigma}_{t+1}^{(s)}$；
- **时间编码更新**：根据最新日历与交易日历，更新 $\boldsymbol{\tau}_{t+1}^{(s)}$；
- **窗口前移**：
  $$
  \mathcal{U}_{t-4:t} \leftarrow \left( \mathcal{U}_{t-4.5:t} \setminus \{ \mathbf{u}_{t-5}^{\text{am}} \} \right) \cup \{ \mathbf{u}_{t}^{\text{pm}} \}, \quad t \leftarrow t + 1
  $$


进一步扩展，考虑：
- 加入注意力机制或Transformer结构处理长序列依赖；
- 引入不确定性估计（如分位数回归或贝叶斯神经网络）；
- 支持多资产联合预测，建模跨市场联动。



---

对于股票603678,其数据:`file_path = "no_git_oic/SH.603678.csv"`
```
数据形状: (38640, 7)

前5行数据:
               datetime   open   high    low  close  volume     amount
0  2025-01-02 09:31:00  30.36  30.47  30.13  30.15   674.0  2040503.0
1  2025-01-02 09:32:00  30.14  30.14  30.01  30.03   491.0  1477167.0
...
38638  2025-08-29 14:59:00  40.50  40.50  40.50  40.50     0.0        0.0
38639  2025-08-29 15:00:00  40.50  40.50  40.50  40.50   887.0  3592350.0

列名: ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']

数据类型:
 datetime     object
open        float64
high        float64
low         float64
close       float64
volume      float64
amount      float64
dtype: object
```

我在想对期日内价格极值（高点/低点）的精准滚动预测,[暂时不考虑黑天鹅事件和市场协同驱动]

设计一个**时序感知 + 统计归一化 + 时间结构编码 + 市场协同驱动 + 滚动自更新**的闭环预测系统。
基于历史分钟级数据，以半日（上午 9:30–11:30，下午 13:00–15:00）为基本预测单元，构建滚动递推预测机制。

首先，对每个半日时段 $ s \in \{ \text{am}, \text{pm} \} $，基于分钟级数据聚合形成原始特征向量：

$$
\mathbf{x}_t^{(s)} = \left[ \text{OHLCV}, \text{RSI}, \text{MACD}, \sigma_{\text{realized}}, \dots \right]^\top \in \mathbb{R}^{d_0}
$$

其中包含价格、成交量、技术指标与波动率等，维度为 $ d_0 $。

为进一步提升模型泛化能力，我们**增强输入特征**，构造：

$$
\mathbf{\tilde{x}}_t^{(s)} = \left[ \mathbf{x}_t^{(s)}; \mathbf{z}_t^{(s)}; \boldsymbol{\tau}_t^{(s)} \right] \in \mathbb{R}^d, \quad d = d_0 + d_0 + d_\tau
$$

该增强向量由三部分拼接而成：

1. **原始行情与技术特征**：$\mathbf{x}_t^{(s)} \in \mathbb{R}^{d_0}$  
   → 包含 OHLCV、RSI、MACD、已实现波动率等，维度为 $ d_0 $

2. **标准化后的 z-score 特征**：$\mathbf{z}_t^{(s)} \in \mathbb{R}^{d_0}$  
   → 与原始特征一一对应，为滚动标准化版本：
   $$
   \mathbf{z}_t^{(s)} = \dfrac{ \mathbf{x}_t^{(s)} - \boldsymbol{\mu}_t^{(s)} }{ \boldsymbol{\sigma}_t^{(s)} }
   $$
   其中均值与标准差基于最近20个交易日（$t-20$ 至 $t-1$）滚动计算，确保统计稳定性。

3. **时间编码特征**：$\boldsymbol{\tau}_t^{(s)} \in \mathbb{R}^{d_\tau}$  
   → 引入结构性时间信息，包含：
   - 星期几（one-hot 编码），
   - 是否月末前3日（布尔标记），
   - 是否节假日前/后1日（布尔标记），
   - 时段标识（am/pm，1维），
   - 月内交易日序号（模周期编码）等。

因此，增强特征总维度为：
> $ d = d_0 + d_0 + d_\tau $

为捕捉市场整体动量与行业联动效应，我们进一步引入外部协变量：

$$
\mathbf{c}_t^{(s)} \in \mathbb{R}^{d_c}
$$

例如：大盘指数收益率、行业动量因子、关联个股特征、资金流指标等。

最终，模型输入向量为：

$$
\mathbf{u}_t^{(s)} = \left[ \mathbf{\tilde{x}}_t^{(s)}; \mathbf{c}_t^{(s)} \right] \in \mathbb{R}^{d + d_c}
$$

预测采用**上午 → 下午 → 滚动更新**的两阶段递推结构，确保信息流与时序一致性。

输入：最近5个交易日（共10个半日）的历史特征序列：

$$
\mathcal{U}_{t-5:t-1} = \left\{ \mathbf{u}_{t-5}^{\text{am}}, \mathbf{u}_{t-5}^{\text{pm}}, \dots, \mathbf{u}_{t-1}^{\text{am}}, \mathbf{u}_{t-1}^{\text{pm}} \right\}
$$

输出：预测上午时段极值：

$$
\hat{y}_t^{\text{am}} = \left( \hat{H}_t^{\text{am}}, \hat{L}_t^{\text{am}} \right) = f_\theta \left( \mathcal{U}_{t-5:t-1} \right)
$$

在上午预测完成后，将上午时段的真实观测值（或预测值，视策略而定）$\mathbf{u}_t^{\text{am}}$ 加入输入窗口，形成“4.5天 + 0.5天”的滚动序列：

$$
\mathcal{U}_{t-4.5:t} = \left\{ \mathbf{u}_{t-5}^{\text{pm}}, \mathbf{u}_{t-4}^{\text{am}}, \dots, \mathbf{u}_{t-1}^{\text{pm}}, \mathbf{u}_{t}^{\text{am}} \right\}
$$

输出：预测下午时段极值：

$$
\hat{y}_t^{\text{pm}} = \left( \hat{H}_t^{\text{pm}}, \hat{L}_t^{\text{pm}} \right) = f_\theta \left( \mathcal{U}_{t-4.5:t} \right)
$$

完成 $t$ 日预测后，系统执行以下更新操作，确保下一交易日预测的连续性与适应性：

- **数据更新**：将 $t$ 日下午的真实观测 $\mathbf{u}_t^{\text{pm}}$ 加入历史序列；
- **统计参数更新**：滑动窗口移除 $t-20$ 日数据，加入 $t$ 日数据，重新计算 $\boldsymbol{\mu}_{t+1}^{(s)}, \boldsymbol{\sigma}_{t+1}^{(s)}$；
- **时间编码更新**：根据最新日历与交易日历，更新 $\boldsymbol{\tau}_{t+1}^{(s)}$；
- **窗口前移**：
  $$
  \mathcal{U}_{t-4:t} \leftarrow \left( \mathcal{U}_{t-4.5:t} \setminus \{ \mathbf{u}_{t-5}^{\text{am}} \} \right) \cup \{ \mathbf{u}_{t}^{\text{pm}} \}, \quad t \leftarrow t + 1
  $$

我选择什么模型,或者怎么做呢?

---

模型选择：LSTM-based Encoder-Decoder（首选）或Transformer（备选）
   - **为什么LSTM？**
     - 时序感知强：LSTM擅长处理序列依赖（如前半日影响后半日），并通过门机制（遗忘/输入/输出门）捕捉市场动量和波动聚类。
     - 适合小样本：您的历史数据约200个交易日（≈400半日），LSTM泛化好，不易过拟合。
     - 闭环友好：易集成滚动更新，支持在线预测（无需全量重训）。
     - 输出回归：最后一层全连接（FC）层直接回归H/L，损失函数用MSE（均方误差）或Huber Loss（对异常值鲁棒）。
     - 扩展性：可加注意力机制（LSTM+Attention）模拟市场协同。

   - **备选：Transformer（如果计算资源充足）**
     - 优势：自注意力捕捉全局依赖（如跨日联动），并行计算快。适合如果d_c增大（更多外部协变量）。
     - 缺点：需位置编码（您的τ已部分覆盖），数据少时需Dropout/正则化防过拟合。
     - 何时选：**如果LSTM验证MAE>2%（价格尺度），切换Transformer。**


Rolling Extrema Predictor

---

- 按交易日滚动,滑动10步预测11
     - 对于第 `k` 天（`k >= 5`）：用过去5天（10个半日：从 `(k-5)` 天 AM 到 `(k-1)` 天 PM）预测第 `k` 天 AM 的 high 和 low
     - 然后，加入真实第 `k` 天 AM，形成新窗口（从 `(k-5)` 天 PM 到第 `k` 天 AM，10个半日），预测第 `k` 天 PM 的 high 和 low
   - 输入 `x`：形状 `(10, feature_dim)`，不包含目标
   - 目标 `y`：形状 `(2,)`，即 `[high, low]`（浮点）


---

模型似乎有问题,不管什么预测的数据似乎都一样
```
2025-09-24 10:34:42,771 | __main__ | INFO | train_model:47 | Epoch 9, Batch 150, Output sample: [42.058865 40.808727]
2025-09-24 10:34:43,149 | __main__ | INFO | train_model:47 | Epoch 9, Batch 200, Output sample: [43.678802 42.33462 ]
2025-09-24 10:34:43,529 | __main__ | INFO | train_model:47 | Epoch 9, Batch 250, Output sample: [43.498688 42.1887  ]
2025-09-24 10:34:43,552 | __main__ | INFO | train_model:53 | Epoch [9/10], Avg Loss: 324.4628
2025-09-24 10:34:43,555 | __main__ | INFO | train_model:62 | Loss not improving, consider early stop.
2025-09-24 10:34:43,565 | __main__ | INFO | train_model:47 | Epoch 10, Batch 0, Output sample: [43.642357 42.35166 ]
...
2025-09-24 10:34:45,448 | __main__ | INFO | train_model:47 | Epoch 10, Batch 250, Output sample: [43.612827 42.31178 ]
2025-09-24 10:34:45,470 | __main__ | INFO | train_model:53 | Epoch [10/10], Avg Loss: 327.1415
2025-09-24 10:34:45,604 | __main__ | INFO | evaluate_model:78 | Batch 21 Pred: [43.9082   42.635834], Target: [38.08 37.21]
2025-09-24 10:34:45,636 | __main__ | INFO | evaluate_model:78 | Batch 27 Pred: [43.9082   42.635834], Target: [40.99 39.6 ]
...
Prediction              Actual
----------------------------------------
43.9082, 42.6358                39.9700, 38.6300
...
43.9082, 42.6358                40.0900, 37.3800
```

- model.py
```python
class LSTMPredictor(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=64, num_layers=2, output_dim=2, dropout=0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            dropout=self.dropout,  # 在 LSTM 中，dropout 只在层与层之间应用，不在时间步之间应用
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(
            x
        )  # encoder: 取最后hidden, hn[-1]==>(batch_size, hidden_dim)
        out = self.fc(hn[-1])  # decoder: 全连接层 FC 回归
        return out
		
import os
import sys
from dotenv import load_dotenv
from termcolor import colored
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from z_utils.logging_config import get_logger
from dataset.rep_dataset import RollingExtremaDataset
from model.rep_lstm_model import LSTMPredictor

load_dotenv()
logger = get_logger(__name__)


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_loss = float("inf")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # 梯度裁剪，防爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

            # 每 k batch 打印一个输出样例
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx}, Output sample: {outputs[0].detach().cpu().numpy()}"
                )

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        logger.info(
            colored(
                f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}", "green"
            )
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
        else:
            logger.info("Loss not improving, consider early stop.")


def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

            # 打印每个 batch 的第一个样例
            logger.info(
                colored(
                    f"Batch {batch_idx} Pred: {outputs[0].numpy()}, Target: {y_batch[0].numpy()}",
                    "blue",
                )
            )

    avg_loss = total_loss / len(test_loader)
    logger.info(colored(f"Test Avg MSE: {avg_loss:.4f}", "blue"))

    # 合并打印前 5 个预测 vs 实际（表格形式）
    preds_flat = np.concatenate(all_preds)[:5]
    targets_flat = np.concatenate(all_targets)[:5]
    print("Prediction\t\tActual")
    print("-" * 40)
    for p, t in zip(preds_flat, targets_flat):
        print(f"{p[0]:.4f}, {p[1]:.4f}\t\t{t[0]:.4f}, {t[1]:.4f}")

    return avg_loss


def main():
    """
    主函数：加载数据、训练模型、评估并保存
    """
    features = 6
    file_path = "no_git_oic/"
    batch_size = 4
    num_epochs = 10
    hidden_dim = 64
    num_layers = 2
    output_dim = 2
    dropout = 0.2
    learning_rate = 0.003
    split_ratio = 0.9

    # 加载数据集
    train_dataset = RollingExtremaDataset(
        file_path, split="train", split_ratio=split_ratio
    )
    test_dataset = RollingExtremaDataset(
        file_path, split="test", split_ratio=split_ratio
    )

    logger.info(colored(f"Train dataset size: {len(train_dataset)}", "yellow"))
    logger.info(colored(f"Test dataset size: {len(test_dataset)}", "yellow"))

    sample_x, sample_y = train_dataset[0]
    logger.info(colored(f"Train x.shape: {sample_x.shape}", "yellow"))
    logger.info(colored(f"Train y: {sample_y}", "yellow"))

    sample_x_test, sample_y_test = test_dataset[0]
    logger.info(colored(f"Test x.shape: {sample_x_test.shape}", "yellow"))
    logger.info(colored(f"Test y: {sample_y_test}", "yellow"))

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = LSTMPredictor(features, hidden_dim, num_layers, output_dim, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练
    logger.info(colored("Starting training...", "green"))
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # 评估
    logger.info(colored("Starting evaluation...", "blue"))
    evaluate_model(model, test_loader, criterion)

    # 保存模型
    os.makedirs("no_git_oic/models", exist_ok=True)
    torch.save(model.state_dict(), "no_git_oic/models/lstm_predictor.pth")
    logger.info(colored("Model saved to no_git_oic/models", "magenta"))


if __name__ == "__main__":
    main()
```

---

- 如果 `RollingExtremaDataset` 生成的所有样本 `x`（序列）内容高度相似（例如，所有滚动窗口的极值数据恒定，或数据文件本身是常量/低方差），LSTM 的隐藏状态 `hn` 会趋于相同，导致 FC 层输出固定。
- 训练中轻微波动可能是 dropout 引起的随机性，但评估时 `model.eval()` 关闭 dropout，输出就“冻结”了。
- 证据：评估中不同 Batch 的 Target 变化（38.08/37.21 → 40.99/39.6），但 Pred 固定；Loss 高但不降，暗示输入无信息。

---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---
