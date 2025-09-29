### 前言

无论是实盘、模拟还是回测，核心都离不开“交易”。看似简单的“交易”二字，背后却关联着众多概念和复杂的运行逻辑。在 Backtrader 中，交易流程大致如下：

1. **设置交易条件**：包括初始资金、交易税费、滑点、成交量限制等；
2. **下达交易指令**：在 Strategy 策略逻辑中调用 `buy`、`sell`、`close` 或 `cancel` 指令；
3. **订单解读**：Order 模块解析交易订单，并将信息传递给经纪商 Broker 模块；
4. **订单检查**：经纪商 Broker 根据订单信息检查并决定是否接受订单；
5. **撮合成交**：经纪商 Broker 接收订单后，按要求撮合成交 trade 并进行结算；
6. **返回执行结果**：Order 模块返回经纪商 Broker 的订单执行结果。

### 回顾一下

`Backtrader` 以大脑 `cerebro` 为统一的调度中心，数据、策略、回测条件等信息都会导入 `cerebro` 中，并由 `cerebro` 启动和完成回测，最后返回回测结果

##### 选股回测流程示例:

实例化大脑 → 导入数据 → 配置回测条件 → 编写交易逻辑 → 运行回测 → 提取回测结果 

##### Backtrader架构:
```
Cerebro
├── DataFeeds (数据模块)
│   ├── CSVDataBase 导入CSV
│   ├── PandasData 导入 df
│   └── YahooFinanceData 导入网站数据 ...
├── Strategy (策略模块)
│   ├── next() 主策略函数
│   ├── notify_order、notify_trade 打印订单、交易信息 ...
├── Indicators (指标模块)
│   ├── SMA、EMA 移动均线
│   └── Ta-lib 技术指标库 ...
├── Orders (订单模块)
│   ├── buy() 买入
│   ├── sell() 卖出
│   ├── close() 平仓
│   └── cancel() 取消订单 ...
├── Sizers (仓位模块)
├── Broker (经纪商模块)
│   ├── cash 初始资金
│   ├── commission 手续费
│   └── slippage 滑点 ...
├── Analyzers (策略分析模块)
│   ├── AnnualReturn 年化收益
│   ├── SharpeRatio 夏普比率
│   ├── DrawDown 回撤
│   └── PyFolio 分析工具 ...
└── Observers (观测器模块)
    ├── Broker 资金\市值曲线
    ├── Trades 盈亏曲线
    └── BuySell 买卖点
```

### 成交量限制管理

在 Backtrader 的默认设置下，Broker 在撮合成交订单时，不会将订单上的购买数量与成交当天 bar 的总成交量（volume）进行对比。即使购买数量超出了当天该标的的总成交量，也会按购买数量全部撮合成交。这种“无限的流动性”模式虽然简化了回测，但显然不现实，会导致回测结果与真实交易产生较大偏差。

为了更贴近真实市场，Backtrader 提供了 `fillers` 模块来限制实际成交量。`fillers` 会指导 Broker 在各个成交时间点决定成交多少量。目前有 3 种形式，每种形式都有明确的成交规则：订单实际成交量取（限制量、订单要求的成交数量、当天 volume）中的最小值。如果无法全部成交，则只部分成交剩余数量，并在次日取消未执行部分。


#### 形式1：`bt.broker.fillers.FixedSize(size)`

通过 `FixedSize(size)` 设置最大的固定成交量 `size`。规则如下：
- 实际成交量 = min(size, 当天 volume, 订单要求数量)
- 如果订单要求数量 > 实际成交量，则部分成交，剩余在次日取消。

**设置方式：**
```python
# 通过 BackBroker() 类直接设置
cerebro = Cerebro()
filler = bt.broker.fillers.FixedSize(size=3000)
newbroker = bt.broker.BrokerBack(filler=filler)
cerebro.broker = newbroker

# 或通过 set_filler 方法
cerebro = Cerebro()
cerebro.broker.set_filler(bt.broker.fillers.FixedSize(size=3000))
```

**示例输出（买入 2000 股，FixedSize=3000）：**
- 2025-01-17：volume=869 < 2000 < 3000，只买入 869 股，Remsize=1131（次日取消）。
- 2025-02-22：volume=1627 < 2000 < 3000，只买入 1627 股，Remsize=373（次日取消）。
- 2025-03-15：2000 < 3000 < volume=4078，全成交，Remsize=0。
- 2025-05-20（卖出）：仓位=2000 > volume=1686，只卖出 1686 股，Remsize=-314；次日（2025-06-13）平仓剩余 314 股。

**策略代码片段：**
```python
self.order = self.buy(size=2000)  # 每次买入 2000 股
```
**部分日志：**
```
2025-01-17, BUY EXECUTED, ref:2456, Price: 32.6331, Size: 869.00, Remsize: 1131.00, Cost: 28358.1410, Stock: 600466.SH
2025-02-22, BUY EXECUTED, ref:2458, Price: 34.9155, Size: 1627.00, Remsize: 373.00, Cost: 56807.5806, Stock: 600466.SH
2025-03-15, BUY EXECUTED, ref:2460, Price: 41.2077, Size: 2000.00, Remsize: 0.00, Cost: 82415.4753, Stock: 600466.SH
2025-05-20, SELL EXECUTED, ref:2465, Price: 39.7735, Size: -1686.00, Remsize: -314.00, Cost: 78628.8050, Stock: 600466.SH
2025-06-13, SELL EXECUTED, ref:2466, Price: 40.0948, Size: -314.00, Remsize: 0.00, Cost: 14643.7988, Stock: 600466.SH
```

**注意现象：**
1. 未成交剩余数量不会自动延续到次日，而是次日取消。
2. 部分成交后，次日打印 `notify_order` 取消剩余订单。

#### 形式2：`bt.broker.fillers.FixedBarPerc(perc)`

通过 `FixedBarPerc(perc)` 将当天 volume 的 `perc%` 作为最大成交量（perc 范围 [0.0, 100.0]）。规则如下：
- 实际成交量 = min(volume * perc / 100, 订单要求数量)
- 部分成交规则同上。

**设置方式：**
```python
# 通过 BackBroker() 类直接设置
cerebro = Cerebro()
filler = bt.broker.fillers.FixedBarPerc(perc=50)
newbroker = bt.broker.BrokerBack(filler=filler)
cerebro.broker = newbroker

# 或通过 set_filler 方法
cerebro = Cerebro()
cerebro.broker.set_filler(bt.broker.fillers.FixedBarPerc(perc=50))
```

**示例输出（买入 2000 股，perc=50%）：**
- 2025-01-17：2000 > volume=869 * 50% = 434.5，只买入 434 股，Remsize=1566（次日取消）。
- 2025-02-22：类似，只买入 813.5 股，Remsize=1187。
- 2025-03-15：2000 < volume=4078 * 50% = 2039，全成交，Remsize=0。
- 2025-04-03（卖出）：仓位=2000 > volume=3826 * 50% = 1913，只卖出 1913 股，Remsize=-87；次日（2025-05-20）平仓剩余 87 股。

**策略代码片段：**
```python
self.order = self.buy(size=2000)
```
**部分日志：**
```
2025-01-17, BUY EXECUTED, ref:2664, Price: 32.6331, Size: 434.00, Remsize: 1566.00, Cost: 14162.7540, Stock: 600466.SH
2025-02-22, BUY EXECUTED, ref:2666, Price: 34.9155, Size: 813.00, Remsize: 1187.00, Cost: 28386.3325, Stock: 600466.SH
2025-03-15, BUY EXECUTED, ref:2668, Price: 41.2077, Size: 2000.00, Remsize: 0.00, Cost: 82415.4753, Stock: 600466.SH
2025-04-03, SELL EXECUTED, ref:2671, Price: 47.0681, Size: -1913.00, Remsize: -87.00, Cost: 88271.1688, Stock: 600466.SH
2025-05-20, SELL EXECUTED, ref:2672, Price: 39.7735, Size: -87.00, Remsize: 0.00, Cost: 4014.4233, Stock: 600466.SH
```

#### 形式3：`bt.broker.fillers.BarPointPerc(minmov=0.01, perc=100.0)`

该形式考虑价格波动区间（low ~ high），通过 `minmov` 均匀划分价格区间和 volume。规则如下：
- part = floor((high - low + minmov) / minmov)
- volume_per = floor(volume / part)
- 实际成交量 = min(volume_per * perc / 100, 订单要求数量)

**设置方式：**
```python
# 通过 BackBroker() 类直接设置
cerebro = Cerebro()
filler = bt.broker.fillers.BarPointPerc(minmov=0.1, perc=50)
newbroker = bt.broker.BrokerBack(filler=filler)
cerebro.broker = newbroker

# 或通过 set_filler 方法
cerebro = Cerebro()
cerebro.broker.set_filler(bt.broker.fillers.BarPointPerc(minmov=0.1, perc=50))
```

**示例输出（买入 2000 股，minmov=0.1, perc=50%）：**
- 2025-01-17：high=32.9415, low=31.8311, part=12；volume_per=72；成交=36 股，Remsize=1964。

**策略代码片段：**
```python
self.order = self.buy(size=2000)
```
**部分日志：**
```
2025-01-17, BUY EXECUTED, ref:2560, Price: 32.6331, Size: 36.00, Remsize: 1964.00, Cost: 1174.7907, Stock: 600466.SH
```

**验证计算：**
- $part = floor((32.9415 - 31.8311 + 0.1) / 0.1) = 12$
- $volume_per = floor(869 / 12) = 72$
- 成交 = min(72 * 50% , 2000) = 36

这些 fillers 能显著提升回测的真实性，建议根据策略需求选择合适形式。

### 交易时机管理


Backtrader 默认采用“当日收盘后下单，次日以开盘价成交”的模式。这种延迟执行能避免未来函数（look-ahead bias），但在某些场景（如 all-in 全仓）下，会因开盘价波动导致资金不足（例如用收盘价计算仓位，次日开盘上涨）。

为应对特殊需求，Backtrader 提供“cheating”模式：
- Cheat-On-Open
- Cheat-On-Close。
这些模式允许当日成交，但会引入偏差，仅适用于特定测试。

#### 默认模式示例
**日志（Send=下单日，Executed=执行日）：**
```
2025-01-16 Send Buy, open 33.00320305
2025-01-17 Buy Executed at price 32.63307367
2025-01-28 Send Sell, open 33.311644199999996
2025-01-29 Sell Executed at price 33.928526500000004
...
```

#### Cheat-On-Open：当日下单，当日开盘价成交
交易逻辑移至 `next_open()`、`nextstart_open()` 或 `prenext_open()` 方法中。

**设置方式：**
```python
# 方式1
cerebro = bt.Cerebro(cheat_on_open=True)

# 方式2
cerebro.broker.set_coo(True)

# 方式3
newbroker = bt.broker.BackBroker(coo=True)
cerebro.broker = newbroker
```

**策略代码片段：**
```python
class TestStrategy(bt.Strategy):
    def next_open(self):
        if self.order:
            self.cancel(self.order)
        if not self.position:
            if self.crossover > 0:  # 10日均线上穿5日
                print('{} Send Buy, open {}'.format(self.data.datetime.date(), self.data.open[0]))
                self.order = self.buy(size=100)
        elif self.crossover < 0:
            self.order = self.close()
```

**示例日志：**
```
2025-01-17 Send Buy, open 32.63307367
2025-01-17 Buy Executed at price 32.63307367
2025-01-29 Send Sell, open 33.928526500000004
2025-01-29 Sell Executed at price 33.928526500000004
...
```

**与默认比较：**
- 下单延迟从次日提前至当日。
- 执行价从次日开盘改为当日开盘，避免资金不足，但引入偏差。

#### Cheat-On-Close：当日下单，当日收盘价成交
交易逻辑仍写在 `next()` 中。

**设置方式：**
```python
# 方式1
cerebro.broker.set_coc(True)

# 方式2
newbroker = bt.broker.BackBroker(coc=True)
cerebro.broker = newbroker
```

**策略代码片段：**
```python
class TestStrategy(bt.Strategy):
    def next(self):
        if self.order:
            self.cancel(self.order)
        if not self.position:
            if self.crossover > 0:
                print('{} Send Buy, close {}'.format(self.data.datetime.date(), self.data.close[0]))
                self.order = self.buy(size=100)
        elif self.crossover < 0:
            self.order = self.close()
```

**示例日志：**
```
2025-01-16 Send Buy, close 32.63307367
2025-01-16 Buy Executed at price 32.63307367
2025-01-28 Send Sell, close 33.86683827
2025-01-28 Sell Executed at price 33.86683827
...
```

**与默认比较：**
- 执行价从次日开盘改为当日收盘。
- 适合测试收盘价计算仓位，但会使用未来数据（收盘价）。

这些 cheating 模式仅用于调试或特殊场景，生产回测仍推荐默认模式以确保公平性。
