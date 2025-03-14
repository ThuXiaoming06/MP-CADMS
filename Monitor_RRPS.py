import torch
import torch.nn as nn
from torchdiffeq import odeint
from torchvision import models

seed=666
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MultiModalFeatureExtractor(nn.Module):
    def __init__(self, img_feat_dim=128, num_feat_dim=64):
        super().__init__()
        self.resnet1 = models.resnet18(pretrained=True)
        self.resnet1.fc = nn.Linear(self.resnet1.fc.in_features, img_feat_dim)

        self.resnet2 = models.resnet18(pretrained=True)
        self.resnet2.fc = nn.Linear(self.resnet2.fc.in_features, img_feat_dim)

        self.mlp = nn.Sequential(
            nn.Linear(1, 32),  # 假设x3是标量
            nn.ReLU(),
            nn.Linear(32, num_feat_dim)

    def forward(self, x1, x2, x3):
        # x1/x2: (B, T, C, H, W)
        # x3: (B, T, 1)
        B, T = x1.shape[:2]

        # 处理图像特征
        img_feat1 = self.resnet1(x1.view(B * T, *x1.shape[2:]))  # (B*T, D)
        img_feat1 = img_feat1.view(B, T, -1)  # (B, T, D_img)

        img_feat2 = self.resnet2(x2.view(B * T, *x2.shape[2:]))
        img_feat2 = img_feat2.view(B, T, -1)

        # 处理数值特征
        num_feat = self.mlp(x3)  # (B, T, D_num)

        return img_feat1, img_feat2, num_feat


class FusionTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.proj = nn.Linear(128 + 128 + 64, d_model)  # 投影到统一维度
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, 3)

    def forward(self, feat1, feat2, feat3):
        # 输入特征均为 (B, T, D)
        combined = torch.cat([feat1, feat2, feat3], dim=-1)
        x = self.proj(combined)  # (B, T, D_model)

        # Transformer处理时序关系
        x = x.permute(1, 0, 2)  # (T, B, D)
        x = self.transformer(x)
        return x.permute(1, 0, 2)  # (B, T, D)


class TimeAwareLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 门控参数
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim, hidden_dim)

        # 时间衰减参数
        self.gamma = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))

    def step(self, x, t_delta, h_prev, c_prev):
        # 文献公式实现
        i = torch.sigmoid(self.W_i(x))
        f = torch.sigmoid(self.W_f(x) + self.beta * t_delta)
        o = torch.sigmoid(self.W_o(x))

        # 时间感知细胞状态更新
        c_candidate = torch.tanh(self.W_c(x))
        c_t = f * c_prev + i * c_candidate * torch.exp(-self.gamma * t_delta)
        h_t = o * torch.tanh(c_t)

        return h_t, c_t

    def forward(self, x_seq, t_deltas):
        # x_seq: (B, T, D)
        # t_deltas: (B, T, 1)
        B, T, _ = x_seq.shape
        h = torch.zeros(B, self.hidden_dim).to(x_seq.device)
        c = torch.zeros(B, self.hidden_dim).to(x_seq.device)

        outputs = []
        for t in range(T):
            h, c = self.step(x_seq[:, t], t_deltas[:, t], h, c)
            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (B, T, H)


class NeuralODE(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        # 修改输出维度为类别数
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, num_classes)  # 直接输出logits
        )

    def forward(self, t, z):
        return self.net(z)


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = MultiModalFeatureExtractor()
        self.fusion = FusionTransformer()
        self.lstm = TimeAwareLSTM(256, 128)

        # 分布参数生成
        self.mu_layer = nn.Linear(128, 64)
        self.logvar_layer = nn.Linear(128, 64)

        self.ode = NeuralODE(64, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x1_seq, x2_seq, x3_seq, t_deltas, pred_times):
        # 特征提取
        f1, f2, f3 = self.feature_extractor(x1_seq, x2_seq, x3_seq)

        fused = self.fusion(f1, f2, f3)  # (B, T, D)

        lstm_out = self.lstm(fused, t_deltas)  # (B, T, H)
        last_hidden = lstm_out[:, -1]
        mu = self.mu_layer(last_hidden)
        logvar = self.logvar_layer(last_hidden)
        z0 = self.reparameterize(mu, logvar)

        sol = odeint(self.ode, z0, pred_times, method='dopri5')
        return sol.permute(1, 0, 2), mu, logvar  # (B, T_pred, C)

num_classes = 2 #
model = FullModel(num_classes)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-3)

for batch in dataloader:
    x1, x2, x3, t_deltas, target_times, target_x3 = batch
    preds, mu, logvar = model(x1, x2, x3, t_deltas, target_times)

    loss = 0
    for t in range(pred_logits.shape[1]):
        loss += criterion(pred_logits[:, t], target_x3[:, t])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()