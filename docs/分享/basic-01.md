## 什么是量化投资？

量化投资就像玩一个“猜价格”的游戏，但我们用数据和公式来猜，而不是靠感觉。比如，买股票时，我们通过历史数据、价格走势等，找到一个大概率赚钱的买卖策略。

- 传统量化：
用固定的数学公式（比如MACD、RSI指标）分析股票。
调整公式里的参数（就像调菜的盐和糖），在历史数据上测试，看能不能赚钱。
缺点：公式是人设计的，可能只在某些市场情况管用，遇到新情况就失效。

- AI量化：
用模型代替固定公式，自动学习复杂的赚钱模式。
比传统量化更灵活，能适应更多市场变化。
比如：输入股票的开盘价、收盘价、交易量等，AI输出“买”或“卖”的建议。

## 量化的门槛在哪里

数学已经不是最大拦路虎了,现有的python库`talib`库封装了大量的技术指标

#### 数据
需要大量历史数据（分钟级、秒级价格等）。
还要过滤“噪音”（没用的信息），比如无关的新闻。

#### 高频交易
赚钱靠速度，比如服务器离交易所近，延迟低到微秒级。
比如Jane Street靠速度赚了印度期权市场70%的利润。
C++开发代码

#### 资金管理
好的AI模型也要会“管钱”。
比如，每次交易用多少钱，亏了怎么止损。

#### 经验
老交易员靠多年经验判断涨跌，新人可以用AI弥补差距。AI没有情绪，能冷静执行策略。


## 常用术语解释

**0. 常用字段**
```
column_mapping = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "change_pct",
    "涨跌额": "change",
    "换手率": "turnover",
    "指数代码": "index_code",
    "指数名称": "index_name",
    "成分券代码": "stock_code",
    "成分券名称": "stock_name",
    "交易所": "exchange",
}
```

**1. 因子**
- **解释**：因子是能够解释资产收益差异的系统性变量。常见因子包括价值（如市盈率）、动量（过去价格趋势）、质量（盈利能力）、波动率、规模（市值）等。
- **典型用途**：构建多因子模型，对股票打分排序

**2. 回测**
- **解释**：利用历史数据模拟交易策略的表现，评估其收益、风险、胜率等指标。回测需注意避免“过拟合”和“前视偏差”（look-ahead bias）。
- **典型用途**：在实盘前验证策略逻辑是否有效

**3. 夏普比率**
- **解释**：衡量单位风险所获得的超额收益，计算公式为：(策略年化收益 - 无风险利率) / 策略年化波动率。越高越好，通常 >1 被认为是较好的策略。
- **典型用途**：比较不同策略的风险调整后收益，是评估策略性价比的核心指标。

**4. 换手率Turnover**
- **解释**：单位时间内资产买卖的频率，通常以年化换手率表示（如年换手率300% = 平均持仓周期约4个月）。高换手意味着高频交易，交易成本更高。
- **典型用途**：用于估算交易成本（佣金、滑点）对策略收益的影响，优化持仓周期。

**5. 滑点（Slippage）**
- **解释**：实际成交价格与预期价格之间的偏差。例如，下单时报价为10元，但因市场流动性不足，实际成交为10.05元，滑点为+0.05元。
- **典型用途**：在回测和实盘中必须考虑滑点，尤其对高频或大额交易策略影响显著。

**6. 复权**
- **解释**：股票在发生分红、送股、配股、拆股等公司行为时，历史价格会出现“跳空”，导致价格序列不连续。**复权**就是对历史价格进行调整，使其反映真实的长期收益，保持价格序列的连续性和可比性。
- **典型用途**：所有基于价格的技术指标（如均线、布林带、RSI）和收益计算**必须使用复权价格**

**7. 超买**
- **解释**：价格短期涨幅过大，可能回调

**8. 超卖**
- **解释**：价格短期跌幅过大，可能反弹

**9. 布林带**
- **解释**：布林带是一种基于价格波动性的技术指标，由三条线组成：
  - 中轨（Middle Band）：通常是 N 日（常用20日）的简单移动平均线（SMA）；
  - 上轨（Upper Band）：中轨 + K 倍（常用2倍）N 日价格的标准差；
  - 下轨（Lower Band）：中轨 - K 倍 N 日价格的标准差。  
价格在大多数时间会围绕均值波动，当价格触及上轨或下轨时，可能预示着短期超买或超卖。
- **典型用途**：上下轨之间的宽度（Band Width）直接反映市场波动率。带宽收窄（“布林带收口”）常预示即将出现大幅波动，可用于触发波动率突破策略

**10. SMA--Simple Moving Average，简单移动平均线**
- **解释**：SMA 是过去 **N 个交易日收盘价的算术平均值**。它对所有历史价格**一视同仁**，权重相同
- **典型用途**：长期趋势、稳健策略

**11. EMA--Exponential Moving Average，指数移动平均线**
- **解释**：EMA 也是一种移动平均，但它**给近期价格赋予更高权重**，对最新价格变化更敏感，能更快反映趋势转折
EMA 采用递推方式计算：
$$
\text{EMA}_t = \alpha \cdot P_t + (1 - \alpha) \cdot \text{EMA}_{t-1}
$$
其中：
- $ P_t $ 是当日收盘价；
- $ \alpha = \frac{2}{N + 1} $ 是平滑系数（如 N=12，则 $ \alpha ≈ 0.1538 $）；
- 初始值通常设为前 N 日的 SMA。
- **典型用途**：
    - 趋势判断
        - **价格 > MA**：视为多头趋势；
        - **价格 < MA**：视为空头趋势。  
        → 常用 EMA（如 20日、50日）用于日内或短线策略，SMA（如 200日）用于长期趋势过滤。
    - 均线交叉策略
        - **金叉**：短期 MA 上穿长期 MA → 买入信号；
        - **死叉**：短期 MA 下穿长期 MA → 卖出信号。  
        → **EMA 交叉反应更快**，适合短线；**SMA 交叉更稳**，适合中长线。

**12. MACD--Moving Average Convergence Divergence，指数平滑异同移动平均线**
- **解释**：MACD 是一种**趋势跟踪型动量指标**，通过比较不同周期的指数移动平均线（EMA）来判断价格的**趋势方向、强度和潜在转折点**。
它由三部分组成：
- **DIF（快线）**：短期 EMA 与长期 EMA 的差值（常用 12 日 EMA - 26 日 EMA）；这个差值本质上衡量的是短期趋势相对于长期趋势的偏离程度。因为 DIF 直接由两个 EMA 相减得出，它会迅速响应价格变化——只要短期 EMA 开始加速或减速，DIF 就会立刻变动。所以 DIF 对市场动量的变化非常敏感，因此被称为“快线”。
- **DEA（慢线/信号线）**：DIF 的 9 日 EMA，用于平滑 DIF；由于 DEA 是 DIF 的平滑版本，它的变化总是滞后于 DIF，走势更平稳、更“迟钝”，因此被称为“慢线”或“信号线”。
- **MACD 柱（Histogram）**：DIF 与 DEA 的差值（即 DIF - DEA），用柱状图表示，反映动量加速或减速。
- **典型用途**：当短期趋势强于长期趋势时，DIF 上穿 DEA，视为看涨信号；反之则看跌

**13. RSI--Relative Strength Index，相对强弱指数**
- **解释**：是一种**动量振荡器（Oscillator）**，用于衡量价格变动的速度和幅度，判断资产是否处于**超买或超卖状态**,价格上涨时，收盘价普遍高于前一日；下跌时则相反。RSI 通过比较上涨日与下跌日的平均涨幅/跌幅，量化这种“相对强度”。
`Δ = 收盘价_t - 收盘价_{t-1}`
计算 N 日:`U = avg(所有上涨日的Δ)`，`D = avg(所有下跌日的|Δ|)`
$RS = U / D$, $RSI = 100 - (100 / (1 + RS))$
    - **RSI > 70**：超买（可能回调）
    - **RSI < 30**：超卖（可能反弹）
- **典型用途**：MACD 金叉 + RSI 从30回升 → 提高买入信号胜率,  MACD 死叉 + RSI > 70 → 强化卖出信号

**14. 中证500成分股**
  中证500成分股是指中证500指数的构成股票，由中证指数有限公司编制，选取A股市场中剔除沪深300指数成分股及总市值排名前300的股票后，总市值排名靠前的500只中小市值股票组成。

**15. 交易税费**
- **解释**：目前 A 股的交易费用分为 2 部分：佣金和印花税。
  - **佣金**：双边征收，不同证券公司收取的佣金各不相同，一般在 0.02%-0.03% 左右，单笔佣金不少于 5 元。
  - **印花税**：只在卖出时收取，税率为 0.1%。


## 常用免费数据源使用介绍

```shell
uv pip install akshare yfinance
```

### akshare

AKShare 是一个专注于中国金融市场的开源 Python 库，提供股票、期货、期权、基金、债券、宏观经济、新闻舆情等多维度数据，数据来源包括交易所、统计局、央行、新浪、东方财富等公开渠道。其优势在于覆盖全面、更新及时、完全免费，且对中文用户友好。

容易封ip

- 最常用
```python
    df = ak.stock_zh_a_hist_min_em(
        symbol=symbol,
        period=period,
        adjust=adjust,
        start_date=start_time,
        end_date=end_time,
    )

    # 列重命名映射
    column_mapping = {
        "时间": "datetime",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
```

```python
import akshare as ak

# 获取贵州茅台（600519）日线数据
stock_data = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date="20230101", end_date="20240101")
print(stock_data.head())
```

```python
# 中国季度GDP
gdp_data = ak.macro_china_gdp()
print(gdp_data)
```

```python
# 易方达蓝筹精选混合（005827）
fund_data = ak.fund_open_fund_info_em(fund="005827", indicator="单位净值走势")
print(fund_data)
```


### yfinance

yfinance 是一个基于 Yahoo Finance 的非官方 Python 接口，主要用于获取全球股票、指数、ETF、外汇、加密货币等金融数据。其优势在于支持多市场（美股、港股、A股等）、历史数据完整、支持复权处理，且使用简单。

```python
    df = yf.download(
        tickers=ticker,
        start=start_date_fmt,
        end=end_date_fmt,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    df.rename(
        columns={
            "Datetime": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )
```
```python
# 贵州茅台（A股代码：600519.SS 表示上交所，.SZ 表示深交所）
maotai = yf.Ticker("600519.SS")
maotai_hist = maotai.history(start="2023-01-01", end="2024-01-01")
print(maotai_hist.head())
```
```python
# 标普500指数
sp500 = yf.Ticker("^GSPC")
sp500_hist = sp500.history(period="6mo")

# 黄金ETF (GLD)
gld = yf.Ticker("GLD")
gld_hist = gld.history(period="1y")
```

```python
# 获取公司基本信息
info = aapl.info
print(f"公司名称: {info['longName']}")
print(f"市值: {info['marketCap']}")
print(f"市盈率: {info.get('trailingPE', 'N/A')}")
```

## 常用回测框架介绍

```
+-------------------+                       +-------------------+
| 1. 准备回测数据   |                       | 4. 设置回测参数   |
+-------------------+                       +-------------------+
          \                                       /
           \                                     /
            v                                   v
+-------------------+                       +-------------------+
| 2. 编写策略       |                       | 5. 设置绩效分析指标|
+-------------------+                       +-------------------+
          \                                       /
           \                                     /
            v                                   v
                    +-------------------+  
                    | 3. 实例化          |  
                    | cerebro =Cerebro()|   
                    +-------------------+  
                            |                                       
                            |                                     
                            v                                   
                    +-------------------+                       
                    | 6. 运行回测        |                      
                    | cerebro.run()     |                     
                    +-------------------+        
                            |                                       
                            |                                     
                            v                                   
                    +-------------------+   
                    | 7. 获得回测结果    |   
                    +-------------------+  
```

实例化大脑 → 导入数据 → 配置回测条件 → 编写交易逻辑 → 运行回测 → 提取回测结果 

```python
import backtrader as bt
import backtrader.indicators as btind

# 创建策略
class TestStrategy(bt.Strategy):
    params = (
        ('maperiod', 20),  # 移动均线周期，示例设为20
    )

    def log(self, txt, dt=None):
        '''打印日志'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        '''初始化属性和指标'''
        # 计算简单移动均线
        self.sma = btind.SimpleMovingAverage(self.datas[0].close, period=self.params.maperiod)
        # 跟踪订单状态
        self.order = None

    def notify_order(self, order):
        '''处理订单状态'''
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行, 价格: {order.executed.price:.2f}, 数量: {order.executed.size}')
            elif order.issell():
                self.log(f'卖出执行, 价格: {order.executed.price:.2f}, 数量: {order.executed.size}')
            self.order = None  # 重置订单状态

    def next(self):
        '''交易逻辑'''
        if self.order:  # 检查是否有未完成订单
            return

        # 示例策略：价格突破均线买入，跌破均线卖出
        if not self.position:  # 没有持仓
            if self.datas[0].close[0] > self.sma[0]:
                self.order = self.buy(size=100)  # 买入100股
                self.log('发出买入信号')
        else:
            if self.datas[0].close[0] < self.sma[0]:
                self.order = self.sell(size=100)  # 卖出100股
                self.log('发出卖出信号')

# 实例化Cerebro引擎
cerebro = bt.Cerebro()

# 加载数据
data = bt.feeds.YahooFinanceCSVData(
    dataname='data.csv',
    fromdate=datetime.datetime(2020, 1, 1),
    todate=datetime.datetime(2023, 12, 31)
)
cerebro.adddata(data)

# 通过经纪商设置初始资金
cerebro.broker.setcash(100000.0)  # 初始资金10万

# 设置交易单位（固定100股）
cerebro.addsizer(bt.sizers.FixedSize, stake=100)

# 设置佣金（示例：0.1%）
cerebro.broker.setcommission(commission=0.001)

# 添加策略
cerebro.addstrategy(TestStrategy)

# 添加分析指标（如夏普比率、年化收益率）
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

# 运行回测
results = cerebro.run()

# 打印分析结果
strat = results[0]
print(f"夏普比率: {strat.analyzers.sharpe.get_analysis()['sharperatio']:.2f}")
print(f"最大回撤: {strat.analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")
print(f"总收益率: {strat.analyzers.returns.get_analysis()['rtot']*100:.2f}%")

# 可视化结果
cerebro.plot()
```

## 回测演示

##### 策略1说明

0. 固定好最开始的 500 支股票不变
1. 按收益率降序排序，最开始选择 2023-01 收益前 20% 的股票
2. 在每月最后一个交易日，计算成分股上个月的收益率(使用**前复权数据**)
3. 在每月第一个交易日，以开盘价清仓旧持仓并买入新选股
4. 持仓权重根据上月收益率加权数据分配
5. 考虑 0.03% 双边佣金和 0.01% 双边滑点
6. 添加年化收益率,交易次数/换手率,夏普比率,最大回撤和总回报分析器
7. 输出回测结果并可视化净值曲线

| 股票池         | 中证 500 成分股。 |
|----------------|--------------------|
| 回测区间       | 2023-02-01 至 2025-06-01。 |
| 持仓周期       | 月度调仓，每月第一个交易日，以开盘价买入或卖出。 |
| 持仓权重       | 上月收益率加权 |
| 总资产         | 100,000,000 元。 |
| 佣金           | 0.0003 双边。 |
| 滑点           | 0.0001 双边。 |
| 策略逻辑       | 每次选择中证 500 成分股中表现最优的前 20% 的股票作为下一个月的持仓成分股，然后在下个月的第一个交易日，卖出已有持仓，买入新的持仓。 |


##### 策略2说明

通过分析大跌后的次日反弹行情，捕捉市场短期反弹机会，实现稳定盈利。

1. 收集最近5年的沪深300ETF（510300）每日行情数据。
2. 筛选出当日收盘价下跌的交易日。
3. 统计次日（即T+1日）的行情表现，包括：
   - 涨跌幅分布
   - 平均收益率
   - 胜率（次日上涨的概率）
   - 最大/最小收益率
4. 基于统计结果，识别具有稳定盈利潜力的操作模式，例如：
   - 确定触发条件（如跌幅超过某阈值）。
   - 在次日开盘价买入，并在收盘价卖出。
5. 制定交易规则，确保策略在回测中考虑实际交易成本。

| 数据源         | 沪深300ETF每日行情数据。 |
|----------------|-----------------------------------------|
| 数据周期       | 最近5年（2020-06-01 至 2025-06-01）。   |
| 数据内容       | 开盘价、收盘价、最高价、最低价、成交量。 |
| 数据类型       | 前复权价格数据。                        |

测试策略一

https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzU2NDAxMDMxOQ==&action=getalbum&album_id=3982551063856562182&scene=126&sessionid=1759022145569#wechat_redirect

## 自己的策略分享

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

演示效果+

```
========================================================================================================================
Prediction (normalized)        Actual (normalized)            Pred (high, low)               Actual (high, low)
========================================================================================================================
0.1241, 0.1029                 0.2145, -0.0587                40.6587, 39.3950               40.7800, 39.2200
0.3342, 0.3346                 0.1400, 0.3385                 40.9407, 39.6458               40.6800, 39.6500
```