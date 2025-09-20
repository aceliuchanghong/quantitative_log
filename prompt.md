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



---




---



---




---
