"""
================================================================================
Trans_TSMonitor_eva_detail.py
Evaluation script for the MP-CADMS longitudinal monitoring module (detailed comments).

Overview:
  Load a trained checkpoint and evaluate recurrence/recovery (rr) or severity
  progression (severity) prediction on the test set. Report accuracy, precision,
  recall, F1, and confusion matrix.
================================================================================
"""

# ========================= Imports =========================
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time
import math

# ========================= NeuralODE integrator =========================
# Same as the training script: prefer torchdiffeq; fall back to a placeholder
try:
    from torchdiffeq import odeint
except ImportError:
    def odeint(func, x0, t):
        """Placeholder odeint: no real integration; repeats x0 to keep shapes."""
        t = torch.tensor(t, dtype=torch.float32, device=x0.device)
        return x0.unsqueeze(0).repeat(len(t), *([1] * x0.dim()))

# =====================================================================
#              Model components (must match the training script)
# =====================================================================

class CNNFeatureExtractor(nn.Module):
    """
    ResNet18-based encoder for IFE images or SCI feature maps.

    Input:  (B, C, H, W), C can be 1 or 3
    Output: (B, output_dim)
    """

    def __init__(self, pretrained=False, output_dim=64):
        super().__init__()
        # Compatible with both new and old torchvision APIs
        try:
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
        except Exception:
            backbone = models.resnet18(pretrained=pretrained)
        # Drop the final FC classifier; keep the convolutional backbone
        modules = list(backbone.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        # Project to a shared feature dimension
        self.fc = nn.Linear(backbone.fc.in_features, output_dim)

    def forward(self, x):
        # ResNet18 expects 3 channels; repeat single-channel inputs
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        feat = self.encoder(x).flatten(1)
        return self.fc(feat)


class LabelMLP(nn.Module):
    """
    Encode a scalar historical diagnosis label into a continuous feature vector.
    Input: labels (B,) -> Output: (B, output_dim)
    """

    def __init__(self, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, labels):
        return self.mlp(labels.unsqueeze(-1).float())


class FusionTransformer(nn.Module):
    """
    Fuse two modality features with a small Transformer Encoder.
    Stack a and b as a length-2 sequence; use the first token as the fused feature.
    """

    def __init__(self, embed_dim=64, n_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)

    def forward(self, a, b):
        seq = torch.stack([a, b], dim=0)  # (2, B, D)
        return self.transformer(seq)[0]


class TimeAwareLSTMCell(nn.Module):
    """
    Time-Aware LSTM cell (Baytas et al., KDD 2017).

    Decompose the previous cell memory into short/long-term subspaces; discount
    only the short-term component by elapsed time Δt between visits:

        C^S_{t-1} = tanh(W_d · C_{t-1} + b_d)   # short-term memory
        Ĉ^S_{t-1} = C^S_{t-1} · g(Δt)           # discounted short-term
        C^T_{t-1} = C_{t-1} − C^S_{t-1}         # long-term memory
        C*_{t-1}  = C^T_{t-1} + Ĉ^S_{t-1}       # adjusted previous memory

    Decay: g(Δt) = 1 / log(e + Δt)
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Standard LSTM gate parameters
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Subspace decomposition network
        self.decomp = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def time_decay(delta_t):
        """g(Δt)=1/log(e+Δt): larger Δt weakens the short-term memory effect."""
        return 1.0 / torch.log(math.e + delta_t)

    def forward(self, x_t, delta_t, state):
        """
        x_t: (B, input_dim), delta_t: (B,), state: (h_prev, c_prev)
        Returns: (h, c)
        """
        h_prev, c_prev = state

        # Subspace decomposition + time discounting
        c_short = torch.tanh(self.decomp(c_prev))
        c_short_dis = c_short * self.time_decay(delta_t.unsqueeze(-1).float())
        c_long = c_prev - c_short
        c_adj = c_long + c_short_dis

        # Standard LSTM gates on the adjusted memory C*
        f = torch.sigmoid(self.W_f(x_t) + self.U_f(h_prev))
        i = torch.sigmoid(self.W_i(x_t) + self.U_i(h_prev))
        o = torch.sigmoid(self.W_o(x_t) + self.U_o(h_prev))
        c_tilde = torch.tanh(self.W_c(x_t) + self.U_c(h_prev))
        c = f * c_adj + i * c_tilde
        h = o * torch.tanh(c)
        return h, c


class TimeAwareLSTM(nn.Module):
    """
    Unroll T-LSTM over a full historical visit sequence.

    Input:  feats (B,T,D), times (B,T)
    Output: hidden (B,T,H), last_hidden (B,H)
    """

    def __init__(self, input_dim=64, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = TimeAwareLSTMCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats, times):
        B, T, _ = feats.shape
        device = feats.device
        h = torch.zeros(B, self.hidden_dim, device=device, dtype=feats.dtype)
        c = torch.zeros(B, self.hidden_dim, device=device, dtype=feats.dtype)
        hidden_seq = []
        prev_times = times[:, 0]
        for t in range(T):
            # Step 0: Δt=0; later steps use the gap between consecutive timestamps
            if t == 0:
                delta_t = torch.zeros_like(times[:, t])
            else:
                delta_t = torch.clamp(times[:, t] - prev_times, min=0.0)
            h, c = self.cell(feats[:, t], delta_t, (h, c))
            h = self.dropout(h)
            hidden_seq.append(h)
            prev_times = times[:, t]
        hidden = torch.stack(hidden_seq, dim=1)
        return hidden, hidden[:, -1]


class GaussianLatentGenerator(nn.Module):
    """
    Map the last T-LSTM hidden state to a Gaussian latent used as NeuralODE init.
    In eval mode, use the mean μ (no stochastic sampling).
    """

    def __init__(self, hidden_dim=64, latent_dim=64):
        super().__init__()
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, h):
        mu = self.mu(h)
        log_var = torch.clamp(self.log_var(h), min=-10.0, max=10.0)
        if self.training:
            std = torch.exp(0.5 * log_var)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu  # deterministic at inference
        return z, mu, log_var


class ODEFunc(nn.Module):
    """
    NeuralODE dynamics f(t, x): dx/dt = f(t, x).
    forward must be (t, x) to match the odeint interface.
    """

    def __init__(self, dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, t, x):
        return self.net(x)


class BinaryMLPHead(nn.Module):
    """Recurrence-recovery binary monitoring head (3-layer MLP)."""

    def __init__(self, dim=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class OrdinalRegressionHead(nn.Module):
    """
    Severity ordinal regression head: WP < P(+) < SP(++).
    Outputs cumulative logits P(y>k), then recovers per-class probabilities.
    """

    def __init__(self, dim=64, num_classes=3, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.score = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim // 2, 1)
        )
        # Learnable cutpoints; softplus enforces strictly increasing order
        self.raw_cutpoints = nn.Parameter(torch.arange(num_classes - 1, dtype=torch.float32))

    def ordered_cutpoints(self):
        first = self.raw_cutpoints[:1]
        if self.raw_cutpoints.numel() == 1:
            return first
        increments = torch.nn.functional.softplus(self.raw_cutpoints[1:])
        return torch.cat([first, first + torch.cumsum(increments, dim=0)], dim=0)

    def forward(self, x):
        score = self.score(x)
        cutpoints = self.ordered_cutpoints().to(x.device).view(*([1] * (x.ndim - 1)), -1)
        return score - cutpoints  # (..., K-1)

    @staticmethod
    def logits_to_probs(logits):
        """Recover P(y=0), P(y=1), ..., P(y=K-1) from cumulative probabilities."""
        p_gt = torch.sigmoid(logits)
        probs = [1.0 - p_gt[..., :1]]
        for k in range(1, p_gt.size(-1)):
            probs.append(p_gt[..., k - 1:k] - p_gt[..., k:k + 1])
        probs.append(p_gt[..., -1:])
        return torch.cat(probs, dim=-1).clamp_min(0.0)


class PatientTimeSeriesModel(nn.Module):
    """
    Full longitudinal monitoring model (must match the training script).

    Path: historical multimodal features -> T-LSTM -> Gaussian latent
          -> NeuralODE -> task head -> probabilities
    """

    def __init__(self, feature_dim=64, num_classes=2, task="rr", dropout=0.2):
        super().__init__()
        self.task = task
        self.cnn_large = CNNFeatureExtractor(output_dim=feature_dim)
        self.cnn_small = CNNFeatureExtractor(output_dim=feature_dim)
        self.label_mlp = LabelMLP(output_dim=feature_dim)
        self.fuse_img = FusionTransformer(embed_dim=feature_dim, dropout=dropout)
        self.fuse_all = FusionTransformer(embed_dim=feature_dim, dropout=dropout)
        self.seq_fuse = TimeAwareLSTM(input_dim=feature_dim, hidden_dim=feature_dim, dropout=dropout)
        self.distribution_generator = GaussianLatentGenerator(hidden_dim=feature_dim, latent_dim=feature_dim)
        self.odefunc = ODEFunc(dim=feature_dim, dropout=dropout)
        if task == "rr":
            self.head = BinaryMLPHead(dim=feature_dim, num_classes=num_classes, dropout=dropout)
        else:
            self.head = OrdinalRegressionHead(dim=feature_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, imgs_large, imgs_small, labels, times, future_times, lengths=None):
        """
        Input:
            imgs_large/imgs_small: (B, T, C, H, W)
            labels: (B, T), times: (B, T), future_times: (T_fut,)
            lengths: (B,) or None — true historical lengths before padding.
                     Used to select the last valid T-LSTM hidden state per sample.
        Output:
            probs: (T_fut, B, num_classes)
        """
        B, T, C, H, W = imgs_large.size()
        feats = []
        for t in range(T):
            v1 = self.cnn_large(imgs_large[:, t])  # IFE
            v2 = self.cnn_small(imgs_small[:, t])  # SCI
            v3 = self.label_mlp(labels[:, t])      # label
            v4 = self.fuse_img(v1, v2)
            v5 = self.fuse_all(v4, v3)
            feats.append(v5)
        seq_feats = torch.stack(feats, dim=1)
        # Do NOT use hidden[:, -1] when sequences are padded to different lengths.
        hidden, _ = self.seq_fuse(seq_feats, times)
        if lengths is None:
            last_hidden = hidden[:, -1]
        else:
            lengths = lengths.to(hidden.device).long().clamp(min=1, max=hidden.size(1))
            idx = torch.arange(hidden.size(0), device=hidden.device)
            last_hidden = hidden[idx, lengths - 1]
        x0, mu, log_var = self.distribution_generator(last_hidden)
        pred_states = odeint(self.odefunc, x0, future_times)  # (T_fut, B, D)
        if self.task == "rr":
            logits = self.head(pred_states)
            return torch.softmax(logits, dim=-1)
        ordinal_logits = self.head(pred_states)
        return OrdinalRegressionHead.logits_to_probs(ordinal_logits)


# =====================================================================
#                         Test data loading
# =====================================================================

# Replace with the actual test npz path
loaded_test = np.load(".../Test_TS_RRSP.npz")
test_ife_images = loaded_test['IFE_image']
test_temporal_labels = loaded_test['temporal_label']
test_sci_images = loaded_test['SCI']
test_true_labels = loaded_test['true_label']

# Try to load real visit timestamps (same candidate keys as training)
_time_keys = ['visit_time', 'visit_times', 'time', 'times', 'elapsed_time', 'time_stamp']
_time_key = next((k for k in _time_keys if k in loaded_test.files), None)
test_visit_times = loaded_test[_time_key] if _time_key is not None else None
if test_visit_times is not None:
    print(f"Using irregular visit times from key '{_time_key}'")
else:
    print("Warning: no visit-time key found; falling back to equal visit intervals")

print(f'Loaded test data shapes:')
print(f'  IFE_image: {test_ife_images.shape}')
print(f'  temporal_label: {test_temporal_labels.shape}')
print(f'  SCI: {test_sci_images.shape}')
print(f'  true_label: {test_true_labels.shape}')

# Ensure numpy arrays
test_ife_images = np.array(test_ife_images)
test_temporal_labels = np.array(test_temporal_labels)
test_sci_images = np.array(test_sci_images)
test_true_labels = np.array(test_true_labels)
test_visit_times = np.array(test_visit_times) if test_visit_times is not None else None

transform_batch = transforms.Compose([transforms.ToTensor()])


# =====================================================================
#                         Test Dataset and collate
# =====================================================================

class PatientTimeSeriesDataset(Dataset):
    """
    Test Dataset: each patient returns historical sequence + future ground-truth labels.
    Returns: ife_seq, sci_seq, temp_labels, times, future_times, true_label
    """

    def __init__(self, ife_images, sci_images, temporal_labels, true_labels, visit_times=None, transform=None):
        self.ife_images = ife_images
        self.sci_images = sci_images
        self.temporal_labels = temporal_labels
        self.true_labels = true_labels
        self.visit_times = visit_times
        self.transform = transform

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, idx):
        ife_seq = self.ife_images[idx]
        sci_seq = self.sci_images[idx]
        temp_labels = self.temporal_labels[idx]
        true_label = self.true_labels[idx]

        # Normalize image layout to (T, C, H, W)
        if ife_seq.ndim == 4:
            if ife_seq.shape[-1] == 3 or ife_seq.shape[-1] == 1:
                ife_seq = np.transpose(ife_seq, (0, 3, 1, 2))
        elif ife_seq.ndim == 3:
            ife_seq = ife_seq[:, np.newaxis, :, :]

        if sci_seq.ndim == 4:
            if sci_seq.shape[-1] == 3 or sci_seq.shape[-1] == 1:
                sci_seq = np.transpose(sci_seq, (0, 3, 1, 2))
        elif sci_seq.ndim == 3:
            sci_seq = sci_seq[:, np.newaxis, :, :]

        ife_seq = torch.from_numpy(ife_seq).float()
        sci_seq = torch.from_numpy(sci_seq).float()
        temp_labels = torch.from_numpy(temp_labels).long()

        if true_label.ndim == 0:
            true_label = np.array([true_label])
        true_label = torch.from_numpy(true_label).long()

        # Historical timestamps: prefer real visit times; else equal spacing
        T_hist = len(temp_labels)
        T_future = len(true_label)
        if self.visit_times is not None:
            times = torch.from_numpy(np.asarray(self.visit_times[idx], dtype=np.float32))
        else:
            times = torch.arange(0, T_hist, dtype=torch.float32)
        future_times = torch.arange(T_hist, T_hist + T_future, dtype=torch.float32)

        return ife_seq, sci_seq, temp_labels, times, future_times, true_label


def my_collate(batch):
    """
    Zero-pad variable-length sequences in a batch (history length may range from
    5 to 33 encounters). Padding is applied at the END of each sequence.

    Also returns ``lengths``: the true (pre-padding) history length of each sample,
    used by the model to select the last valid T-LSTM hidden state.

    Output shapes:
        ife/sci (B, max_T, C, H, W), labels/times (B, max_T),
        true_labels (B, max_T_fut), future_times (T_fut,), lengths (B,)
    """
    ife_seqs, sci_seqs, temp_labels, times, future_times, true_labels = zip(*batch)

    lengths = torch.tensor([len(tl) for tl in temp_labels], dtype=torch.long)

    max_T = max(len(tl) for tl in temp_labels)
    max_T_fut = max(len(fl) for fl in true_labels)

    B = len(batch)
    _, C1, H1, W1 = ife_seqs[0].shape
    _, C2, H2, W2 = sci_seqs[0].shape

    ife_padded = torch.zeros(B, max_T, C1, H1, W1)
    sci_padded = torch.zeros(B, max_T, C2, H2, W2)
    temp_labels_padded = torch.zeros(B, max_T, dtype=torch.long)
    times_padded = torch.zeros(B, max_T)
    true_labels_padded = torch.zeros(B, max_T_fut, dtype=torch.long)

    for i, (ife, sci, tl, t, fl) in enumerate(zip(ife_seqs, sci_seqs, temp_labels, times, true_labels)):
        T = len(tl)
        ife_padded[i, :T] = ife
        sci_padded[i, :T] = sci
        temp_labels_padded[i, :T] = tl
        times_padded[i, :T] = t
        true_labels_padded[i, :len(fl)] = fl

    future_times_vec = future_times[0]

    return ife_padded, sci_padded, temp_labels_padded, times_padded, future_times_vec, true_labels_padded, lengths


# =====================================================================
#                         Build test DataLoader
# =====================================================================

test_dataset = PatientTimeSeriesDataset(
    test_ife_images, test_sci_images, test_temporal_labels, test_true_labels,
    visit_times=test_visit_times, transform=transform_batch
)
# shuffle=False: keep sample order for reproducibility and alignment
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=my_collate)


# =====================================================================
#                         Load model weights
# =====================================================================

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Default task; overridden if checkpoint stores 'task'
task = "rr"
dropout = 0.2
num_classes = 2 if task == "rr" else 3
model = PatientTimeSeriesModel(feature_dim=64, num_classes=num_classes, task=task, dropout=dropout)

# Replace with the checkpoint path saved during training
checkpoint = torch.load('.../Trans_TSMonitor_best.pt', map_location=device)
# Rebuild the model with the stored task to avoid head-structure mismatch
if 'task' in checkpoint:
    task = checkpoint['task']
    num_classes = 2 if task == "rr" else 3
    model = PatientTimeSeriesModel(feature_dim=64, num_classes=num_classes, task=task, dropout=dropout)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()  # disable Dropout; Gaussian latent uses the mean


# =====================================================================
#                         Test-set evaluation
# =====================================================================

print("\n=== Evaluating on Test Set ===")
correct = 0
total = 0
all_preds = []
all_labels = []

start_time = time.time()
with torch.no_grad():  # no gradients needed for evaluation
    for batch in test_loader:
        imgs224, imgs35, labels, times, future_times, future_labels, lengths = batch
        imgs224 = imgs224.to(device)
        imgs35 = imgs35.to(device)
        labels = labels.to(device)
        times = times.to(device)
        future_times = future_times.to(device)
        future_labels = future_labels.to(device)
        lengths = lengths.to(device)

        # preds: (T_fut, B, num_classes)
        # lengths selects the last valid (non-padded) T-LSTM hidden state per patient
        preds = model(imgs224, imgs35, labels, times, future_times, lengths)

        # Flatten over samples x future steps for metric aggregation
        T_fut = preds.size(0)
        B = preds.size(1)
        preds_flat = preds.permute(1, 0, 2).reshape(B * T_fut, -1)
        targets_flat = future_labels.reshape(-1)

        _, predicted = torch.max(preds_flat, 1)  # argmax class
        total += targets_flat.size(0)
        correct += (predicted == targets_flat).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets_flat.cpu().numpy())

end_time = time.time()

# -------------------- Compute and print metrics --------------------
test_acc = 100.0 * correct / total
# macro: equal weight per class (better under class imbalance)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

print(f"\n=== Test Results ===")
print(f"Test Accuracy: {test_acc:.2f}% ({correct}/{total})")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-score: {f1:.4f}")
print(f"Time taken: {end_time - start_time:.2f} seconds")
print(f"\nConfusion Matrix:")
print(cm)
