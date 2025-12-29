import torch
import torch.nn as nn


class StockMixer(nn.Module):
    def __init__(self, n_features, n_time, hidden_dim):
        super(StockMixer, self).__init__()

        # 1. 嵌入层：将原始特征映射到隐藏空间
        self.embedding = nn.Linear(n_features, hidden_dim)

        # 2. Indicator Mixing (作用于特征维度)
        # 实际上在Embedding后，维度已经是hidden_dim，这一步是进一步的特征交互
        self.indicator_mixer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3. Time Mixing (作用于时间维度)
        # 输入形状需要转置，使得Time在最后一维
        self.time_mixer = nn.Sequential(
            nn.Linear(n_time, n_time), nn.ReLU(), nn.Linear(n_time, n_time)
        )

        # 4. Stock Mixing (市场交互)
        # 这里简化为一个市场向量的生成与融合
        self.market_aggregator = nn.Linear(hidden_dim, hidden_dim)
        self.market_distributor = nn.Linear(hidden_dim, hidden_dim)

        # 5. 预测层
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape:
        # 为了简化，我们假设Batch=1，关注 N, T, F

        # --- Step 1: Embedding & Indicator Mixing ---
        # x: ->
        x = self.embedding(x)
        x = self.indicator_mixer(x)  # 混合特征信息

        # --- Step 2: Time Mixing ---
        # 需要转置: ->
        x = x.transpose(1, 2)
        x = self.time_mixer(x)  # 混合时间信息
        # 转置回来: ->
        x = x.transpose(1, 2)

        # --- 聚合时间维度 ---
        # 将T个时间步的信息压缩为一个向量，例如取平均
        # x: -> [N, hidden]
        x_stock = x.mean(dim=1)

        # --- Step 3: Stock Mixing ---
        # 生成市场向量 M: [1, hidden] (对所有股票取平均)
        market_vector = x_stock.mean(dim=0, keepdim=True)
        # 处理市场向量
        market_info = self.market_aggregator(market_vector)

        # 将市场信息融合回个股
        # x_stock: [N, hidden] + market_info: [1, hidden] (广播相加)
        x_final = x_stock + self.market_distributor(market_info)

        # --- Step 4: Prediction ---
        # [N, hidden] -> [N, 1]
        prediction = self.regressor(x_final)

        return prediction.squeeze()
