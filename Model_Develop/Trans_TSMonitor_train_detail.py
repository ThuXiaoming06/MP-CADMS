"""
================================================================================
Trans_TSMonitor_train_detail.py
Training script for the MP-CADMS longitudinal monitoring module (detailed comments).

Overview:
  Given historical IFE images, SCI feature maps, and diagnosis labels from multiple
  patient visits, predict future recurrence/recovery (rr) or severity progression
  (severity).

Pipeline:
  1. Per time step: CNN(IFE) + CNN(SCI) + MLP(label) -> Transformer fusion
  2. Temporal modeling: Time-Aware LSTM (T-LSTM) over irregular visit intervals
  3. Latent: map the last T-LSTM hidden state to a Gaussian; sample NeuralODE init
  4. Future prediction: integrate NeuralODE along future time points
  5. Task head: BinaryMLPHead (binary) or OrdinalRegressionHead (ordinal) -> probs
================================================================================
"""

# ========================= Imports =========================
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold  # patient-level stratified split
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau     # reduce LR when val loss plateaus
from tqdm import tqdm
import pandas as pd
import time
import math

# ========================= NeuralODE integrator =========================
# torchdiffeq.odeint numerically integrates dx/dt = f(t, x) at given time points.
# If torchdiffeq is missing, a shape-preserving placeholder is used for demo/syntax checks.
try:
    from torchdiffeq import odeint
except ImportError:
    def odeint(func, x0, t):
        """Placeholder odeint: no real integration; repeats x0 along the time axis."""
        t = torch.tensor(t, dtype=torch.float32, device=x0.device)
        return x0.unsqueeze(0).repeat(len(t), *([1] * x0.dim()))

# ========================= Reproducibility =========================
seed = 666
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True   # force deterministic conv algorithms
torch.backends.cudnn.benchmark = False      # disable auto-tuner for reproducibility

# =====================================================================
#                         Model components
# =====================================================================

class CNNFeatureExtractor(nn.Module):
    """
    ResNet18-based encoder for IFE images or SCI feature maps.

    Input:  (B, C, H, W), C can be 1 or 3
    Output: (B, output_dim)
    """

    def __init__(self, pretrained=False, output_dim=64):
        super().__init__()
        # Compatible with both new (weights=) and old (pretrained=) torchvision APIs
        try:
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
        except Exception:
            backbone = models.resnet18(pretrained=pretrained)
        # Drop the final FC classifier; keep the convolutional backbone (+ GAP)
        modules = list(backbone.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        # Project ResNet output (typically 512-d) to a shared feature dimension
        self.fc = nn.Linear(backbone.fc.in_features, output_dim)

    def forward(self, x):
        # ResNet18 expects 3-channel input; repeat single-channel inputs
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        feat = self.encoder(x).flatten(1)  # (B, 512)
        return self.fc(feat)               # (B, output_dim)


class LabelMLP(nn.Module):
    """
    Encode a scalar historical diagnosis label into a continuous feature vector.

    Input:  labels of shape (B,)
    Output: (B, output_dim)
    """

    def __init__(self, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, labels):
        # Expand scalar labels to (B, 1) before the MLP
        return self.mlp(labels.unsqueeze(-1).float())


class FusionTransformer(nn.Module):
    """
    Fuse two modality feature vectors with a small Transformer Encoder.
    Stack a and b as a length-2 sequence; use the first token output as the fused feature.

    Input:  a, b each (B, embed_dim)
    Output: (B, embed_dim)
    """

    def __init__(self, embed_dim=64, n_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)

    def forward(self, a, b):
        # seq shape: (2, B, D); TransformerEncoder uses sequence-first by default
        seq = torch.stack([a, b], dim=0)
        return self.transformer(seq)[0]  # first position as fused representation


class TimeAwareLSTMCell(nn.Module):
    """
    Time-Aware LSTM cell (Baytas et al., KDD 2017).

    Unlike a standard LSTM, the previous cell memory is decomposed into short-term
    and long-term subspaces. Only the short-term component is discounted by the
    elapsed time Δt between visits, while the long-term profile is preserved.
    This handles irregular clinical visit intervals (days to months).

    Formulas:
        C^S_{t-1} = tanh(W_d · C_{t-1} + b_d)   # short-term memory
        Ĉ^S_{t-1} = C^S_{t-1} · g(Δt)           # discounted short-term memory
        C^T_{t-1} = C_{t-1} − C^S_{t-1}         # long-term memory
        C*_{t-1}  = C^T_{t-1} + Ĉ^S_{t-1}       # adjusted previous memory
    Standard LSTM gates then operate on C*_{t-1}.

    Decay function (suitable for large medical time gaps):
        g(Δt) = 1 / log(e + Δt)   # larger Δt -> smaller g -> weaker short-term effect
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Standard LSTM gates: forget / input / output / candidate
        # W_* acts on current input x_t; U_* acts on previous hidden state h_{t-1}
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Subspace decomposition network: extract short-term memory from C_{t-1}
        self.decomp = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def time_decay(delta_t):
        """Monotonically non-increasing decay: g(Δt)=1/log(e+Δt)."""
        return 1.0 / torch.log(math.e + delta_t)

    def forward(self, x_t, delta_t, state):
        """
        Args:
            x_t:     current input features, (B, input_dim)
            delta_t: elapsed time since the previous visit, (B,)
            state:   (h_prev, c_prev)
        Returns:
            (h, c): current hidden state and cell memory
        """
        h_prev, c_prev = state

        # ---- Subspace decomposition + time discounting ----
        c_short = torch.tanh(self.decomp(c_prev))                                  # short-term
        c_short_dis = c_short * self.time_decay(delta_t.unsqueeze(-1).float())     # discounted
        c_long = c_prev - c_short                                                  # long-term
        c_adj = c_long + c_short_dis                                               # adjusted C*

        # ---- Standard LSTM gates on C* ----
        f = torch.sigmoid(self.W_f(x_t) + self.U_f(h_prev))   # forget gate
        i = torch.sigmoid(self.W_i(x_t) + self.U_i(h_prev))   # input gate
        o = torch.sigmoid(self.W_o(x_t) + self.U_o(h_prev))   # output gate
        c_tilde = torch.tanh(self.W_c(x_t) + self.U_c(h_prev))  # candidate memory
        c = f * c_adj + i * c_tilde                            # current cell
        h = o * torch.tanh(c)                                  # current hidden
        return h, c


class TimeAwareLSTM(nn.Module):
    """
    Unroll T-LSTM cells over a historical visit sequence.

    Input:
        feats: (B, T, D)  fused features at each time step
        times: (B, T)     visit timestamps (used to compute Δt)
    Output:
        hidden:      (B, T, H)  hidden states for all steps
        last_hidden: (B, H)     last-step hidden state (for downstream modules)
    """

    def __init__(self, input_dim=64, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = TimeAwareLSTMCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats, times):
        B, T, _ = feats.shape
        device = feats.device
        # Initialize hidden state and cell memory to zeros
        h = torch.zeros(B, self.hidden_dim, device=device, dtype=feats.dtype)
        c = torch.zeros(B, self.hidden_dim, device=device, dtype=feats.dtype)
        hidden_seq = []
        prev_times = times[:, 0]
        for t in range(T):
            # First step: Δt=0; later: Δt = current stamp - previous stamp (clamped >= 0)
            if t == 0:
                delta_t = torch.zeros_like(times[:, t])
            else:
                delta_t = torch.clamp(times[:, t] - prev_times, min=0.0)
            h, c = self.cell(feats[:, t], delta_t, (h, c))
            h = self.dropout(h)
            hidden_seq.append(h)
            prev_times = times[:, t]
        hidden = torch.stack(hidden_seq, dim=1)  # (B, T, H)
        return hidden, hidden[:, -1]             # full sequence + last hidden


class GaussianLatentGenerator(nn.Module):
    """
    Map the last T-LSTM hidden state to a Gaussian N(μ, σ²).
    The sampled z is used as the NeuralODE initial condition.

    Training: z = μ + ε·σ  (reparameterization, ε~N(0,I))
    Inference: z = μ       (deterministic)
    """

    def __init__(self, hidden_dim=64, latent_dim=64):
        super().__init__()
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, h):
        mu = self.mu(h)
        # Clamp log_var to avoid numerical overflow
        log_var = torch.clamp(self.log_var(h), min=-10.0, max=10.0)
        if self.training:
            std = torch.exp(0.5 * log_var)
            z = mu + torch.randn_like(std) * std  # reparameterized sample
        else:
            z = mu
        return z, mu, log_var


class ODEFunc(nn.Module):
    """
    NeuralODE dynamics f(t, x) defining continuous latent evolution:
        dx/dt = f(t, x)
    Parameterized by a two-layer MLP.
    Note: forward must be (t, x) to match the odeint interface.
    """

    def __init__(self, dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, t, x):
        # t is the current integration time (unused here; dynamics are time-invariant)
        return self.net(x)


class BinaryMLPHead(nn.Module):
    """
    Recurrence-recovery binary monitoring head.
    Three-layer MLP: dim -> dim -> dim/2 -> num_classes
    """

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
    Severity-progression ordinal regression head.
    Ordered classes: WP < P(+) < SP(++).

    Idea:
      1. Network outputs a scalar severity score
      2. Compare score against learnable ordered cutpoints -> cumulative logits P(y > k)
      3. Convert cumulative probabilities back to per-class probabilities
    """

    def __init__(self, dim=64, num_classes=3, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes
        # Map features to a scalar score
        self.score = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim // 2, 1)
        )
        # Raw cutpoint parameters; softplus enforces strictly increasing cutpoints
        self.raw_cutpoints = nn.Parameter(torch.arange(num_classes - 1, dtype=torch.float32))

    def ordered_cutpoints(self):
        """Convert raw parameters into strictly increasing cutpoints."""
        first = self.raw_cutpoints[:1]
        if self.raw_cutpoints.numel() == 1:
            return first
        # softplus -> positive increments; cumsum -> monotonic cutpoints
        increments = torch.nn.functional.softplus(self.raw_cutpoints[1:])
        return torch.cat([first, first + torch.cumsum(increments, dim=0)], dim=0)

    def forward(self, x):
        score = self.score(x)  # (..., 1)
        # Broadcast cutpoints to match the leading dims of score
        cutpoints = self.ordered_cutpoints().to(x.device).view(*([1] * (x.ndim - 1)), -1)
        return score - cutpoints  # (..., K-1), logits for P(y > k)

    @staticmethod
    def logits_to_probs(logits):
        """
        Recover class probabilities from cumulative logits:
          P(y=0)   = 1 − P(y>0)
          P(y=k)   = P(y>k-1) − P(y>k)
          P(y=K-1) = P(y>K-2)
        """
        p_gt = torch.sigmoid(logits)
        probs = [1.0 - p_gt[..., :1]]
        for k in range(1, p_gt.size(-1)):
            probs.append(p_gt[..., k - 1:k] - p_gt[..., k:k + 1])
        probs.append(p_gt[..., -1:])
        return torch.cat(probs, dim=-1).clamp_min(0.0)


class PatientTimeSeriesModel(nn.Module):
    """
    Full longitudinal monitoring model.

    Forward path:
      historical steps -> CNN(IFE) + CNN(SCI) + MLP(label)
                      -> FusionTransformer
                      -> T-LSTM temporal integration
                      -> Gaussian latent sample as NeuralODE init z0
                      -> odeint to future time points
                      -> task head -> class probabilities

    Args:
        feature_dim: shared feature dimension
        num_classes: number of classes (rr=2, severity=3)
        task:        "rr" (recurrence-recovery) or "severity"
        dropout:     dropout rate
    """

    def __init__(self, feature_dim=64, num_classes=2, task="rr", dropout=0.2):
        super().__init__()
        self.task = task
        # Two CNNs for large (IFE) and small (SCI) images
        self.cnn_large = CNNFeatureExtractor(output_dim=feature_dim)
        self.cnn_small = CNNFeatureExtractor(output_dim=feature_dim)
        self.label_mlp = LabelMLP(output_dim=feature_dim)
        # Two-stage fusion: images first, then with label features
        self.fuse_img = FusionTransformer(embed_dim=feature_dim, dropout=dropout)
        self.fuse_all = FusionTransformer(embed_dim=feature_dim, dropout=dropout)
        # Temporal integration + latent + ODE dynamics
        self.seq_fuse = TimeAwareLSTM(input_dim=feature_dim, hidden_dim=feature_dim, dropout=dropout)
        self.distribution_generator = GaussianLatentGenerator(hidden_dim=feature_dim, latent_dim=feature_dim)
        self.odefunc = ODEFunc(dim=feature_dim, dropout=dropout)
        # Instantiate only the selected task head
        if task == "rr":
            self.head = BinaryMLPHead(dim=feature_dim, num_classes=num_classes, dropout=dropout)
        else:
            self.head = OrdinalRegressionHead(dim=feature_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, imgs_large, imgs_small, labels, times, future_times, lengths=None):
        """
        Args:
            imgs_large:   (B, T, C, H, W)  historical IFE image sequence
            imgs_small:   (B, T, C, H, W)  historical SCI feature-map sequence
            labels:       (B, T)           historical diagnosis labels
            times:        (B, T)           historical visit timestamps
            future_times: (T_fut,)         future prediction time grid (1-D)
            lengths:      (B,) or None     true historical lengths before padding.
                                           If provided, the last valid (non-padded)
                                           T-LSTM hidden state is selected per sample.
                                           If None, falls back to hidden[:, -1].
        Returns:
            probs: (T_fut, B, num_classes) class probabilities at each future time
        """
        B, T, C, H, W = imgs_large.size()
        feats = []
        # Extract and fuse multimodal features per time step
        for t in range(T):
            v1 = self.cnn_large(imgs_large[:, t])   # IFE features
            v2 = self.cnn_small(imgs_small[:, t])   # SCI features
            v3 = self.label_mlp(labels[:, t])       # label features
            v4 = self.fuse_img(v1, v2)              # image fusion
            v5 = self.fuse_all(v4, v3)              # image + label fusion
            feats.append(v5)
        seq_feats = torch.stack(feats, dim=1)       # (B, T, D)
        # T-LSTM over history. Do NOT use hidden[:, -1] when sequences are padded:
        # that index is the padded max length, not each patient's true last visit.
        hidden, _ = self.seq_fuse(seq_feats, times)
        if lengths is None:
            last_hidden = hidden[:, -1]
        else:
            # lengths is 1-based count; index of the last valid step is lengths - 1
            lengths = lengths.to(hidden.device).long().clamp(min=1, max=hidden.size(1))
            idx = torch.arange(hidden.size(0), device=hidden.device)
            last_hidden = hidden[idx, lengths - 1]
        # Gaussian latent -> NeuralODE initial state
        x0, mu, log_var = self.distribution_generator(last_hidden)
        # Integrate along future times -> (T_fut, B, D)
        pred_states = odeint(self.odefunc, x0, future_times)
        # Task head -> probabilities
        if self.task == "rr":
            logits = self.head(pred_states)
            return torch.softmax(logits, dim=-1)
        ordinal_logits = self.head(pred_states)
        return OrdinalRegressionHead.logits_to_probs(ordinal_logits)


# =====================================================================
#                         Data loading
# =====================================================================

# Load training .npz (replace with the actual path).
# Expected keys: IFE_image, temporal_label, SCI, true_label
# Optional keys: visit_time / times / etc. for real visit timestamps
loaded_train = np.load(".../Train_TS_RRSP.npz")
ife_images = loaded_train['IFE_image']            # historical IFE image sequences
temporal_labels = loaded_train['temporal_label']  # historical diagnosis labels
sci_images = loaded_train['SCI']                  # historical SCI feature-map sequences
true_labels = loaded_train['true_label']          # future ground-truth labels (supervision)

# Try to load real (irregular) visit times; fall back to equal spacing if absent
_time_keys = ['visit_time', 'visit_times', 'time', 'times', 'elapsed_time', 'time_stamp']
_time_key = next((k for k in _time_keys if k in loaded_train.files), None)
visit_times = loaded_train[_time_key] if _time_key is not None else None
if visit_times is not None:
    print(f"Using irregular visit times from key '{_time_key}'")
else:
    print("Warning: no visit-time key found; falling back to equal visit intervals")

print(f'Loaded data shapes:')
print(f'  IFE_image: {ife_images.shape}')
print(f'  temporal_label: {temporal_labels.shape}')
print(f'  SCI: {sci_images.shape}')
print(f'  true_label: {true_labels.shape}')

# Ensure numpy arrays
ife_images = np.array(ife_images)
temporal_labels = np.array(temporal_labels)
sci_images = np.array(sci_images)
true_labels = np.array(true_labels)
visit_times = np.array(visit_times) if visit_times is not None else None

# Image transform (ToTensor only here; add normalization etc. if needed)
transform_batch = transforms.Compose([transforms.ToTensor()])

# ---------- Load patient IDs for patient-level split (no patient leakage) ----------
patient_ids = None
try:
    excel_path = "all_sample_information.xlsx"
    df_info = pd.read_excel(excel_path, sheet_name=0)
    # Keep development folds D1–D4 only
    df_d1_d4 = df_info[df_info['Fold'].isin(['D1', 'D2', 'D3', 'D4'])]
    patient_ids_loaded = df_d1_d4['PaID'].values
    if len(patient_ids_loaded) == len(true_labels):
        patient_ids = patient_ids_loaded
        print(f'Loaded {len(patient_ids)} patient IDs matching data length')
    else:
        print(f'Warning: Patient IDs length ({len(patient_ids_loaded)}) does not match data length ({len(true_labels)})')
        print('Using index-based patient IDs (each sample is a separate patient)')
        patient_ids = np.arange(len(true_labels))
except Exception as e:
    print(f'Warning: Could not load patient IDs: {e}')
    print('Using index-based patient IDs (each sample is a separate patient)')
    patient_ids = np.arange(len(true_labels))


# =====================================================================
#                         Dataset and DataLoader
# =====================================================================

class PatientTimeSeriesDataset(Dataset):
    """
    Patient time-series dataset.
    Each sample is one patient's full historical sequence + future labels.

    Returns:
        ife_seq, sci_seq, temp_labels, times, future_times, true_label
    """

    def __init__(self, ife_images, sci_images, temporal_labels, true_labels, visit_times=None, transform=None):
        self.ife_images = ife_images
        self.sci_images = sci_images
        self.temporal_labels = temporal_labels
        self.true_labels = true_labels
        self.visit_times = visit_times  # real visit stamps; None -> equal spacing
        self.transform = transform

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, idx):
        # Fetch data for patient idx
        ife_seq = self.ife_images[idx]
        sci_seq = self.sci_images[idx]
        temp_labels = self.temporal_labels[idx]
        true_label = self.true_labels[idx]

        # ---- Normalize image layout to (T, C, H, W) ----
        # If the last dim is channels (1 or 3), transpose (T,H,W,C) -> (T,C,H,W)
        if ife_seq.ndim == 4:
            if ife_seq.shape[-1] == 3 or ife_seq.shape[-1] == 1:
                ife_seq = np.transpose(ife_seq, (0, 3, 1, 2))
        elif ife_seq.ndim == 3:
            # (T, H, W) -> add channel dim
            ife_seq = ife_seq[:, np.newaxis, :, :]

        if sci_seq.ndim == 4:
            if sci_seq.shape[-1] == 3 or sci_seq.shape[-1] == 1:
                sci_seq = np.transpose(sci_seq, (0, 3, 1, 2))
        elif sci_seq.ndim == 3:
            sci_seq = sci_seq[:, np.newaxis, :, :]

        # Convert to PyTorch tensors
        ife_seq = torch.from_numpy(ife_seq).float()
        sci_seq = torch.from_numpy(sci_seq).float()
        temp_labels = torch.from_numpy(temp_labels).long()

        # Expand scalar labels to length-1 arrays for uniform handling
        if true_label.ndim == 0:
            true_label = np.array([true_label])
        true_label = torch.from_numpy(true_label).long()

        # ---- Build historical timestamps and future time grid ----
        T_hist = len(temp_labels)
        T_future = len(true_label)
        if self.visit_times is not None:
            # Real irregular visit times (T-LSTM uses them to compute Δt)
            times = torch.from_numpy(np.asarray(self.visit_times[idx], dtype=np.float32))
        else:
            # Fallback: equal spacing by index
            times = torch.arange(0, T_hist, dtype=torch.float32)
        # Future times continue after the history (equal spacing; change if needed)
        future_times = torch.arange(T_hist, T_hist + T_future, dtype=torch.float32)

        return ife_seq, sci_seq, temp_labels, times, future_times, true_label


def my_collate(batch):
    """
    Custom collate: zero-pad variable-length sequences in a batch.

    Reason: patients may have different history lengths T (e.g. 5 to 33 encounters);
    default collate cannot stack them. Padding is applied at the END of each sequence.

    Also returns ``lengths``: the true (pre-padding) history length of each sample.
    The model uses lengths to pick the last valid T-LSTM hidden state instead of
    the padded position hidden[:, -1].

    Padded shapes:
        ife/sci:     (B, max_T, C, H, W)
        temp_labels: (B, max_T)
        times:       (B, max_T)
        true_labels: (B, max_T_fut)
        future_times:(T_fut,)  -- taken from the first sample in the batch
        lengths:     (B,)
    """
    ife_seqs, sci_seqs, temp_labels, times, future_times, true_labels = zip(*batch)

    # True history lengths before padding (needed for correct last-hidden selection)
    lengths = torch.tensor([len(tl) for tl in temp_labels], dtype=torch.long)

    max_T = max(len(tl) for tl in temp_labels)       # longest history in batch
    max_T_fut = max(len(fl) for fl in true_labels)   # longest future in batch

    B = len(batch)
    _, C1, H1, W1 = ife_seqs[0].shape
    _, C2, H2, W2 = sci_seqs[0].shape

    # Pre-allocate zero-padded tensors
    ife_padded = torch.zeros(B, max_T, C1, H1, W1)
    sci_padded = torch.zeros(B, max_T, C2, H2, W2)
    temp_labels_padded = torch.zeros(B, max_T, dtype=torch.long)
    times_padded = torch.zeros(B, max_T)
    true_labels_padded = torch.zeros(B, max_T_fut, dtype=torch.long)

    # Copy each sample into the leading portion of the padded tensors
    for i, (ife, sci, tl, t, fl) in enumerate(zip(ife_seqs, sci_seqs, temp_labels, times, true_labels)):
        T = len(tl)
        ife_padded[i, :T] = ife
        sci_padded[i, :T] = sci
        temp_labels_padded[i, :T] = tl
        times_padded[i, :T] = t
        true_labels_padded[i, :len(fl)] = fl

    # Use the first sample's future time grid (assumes aligned future times in the batch)
    future_times_vec = future_times[0]

    return ife_padded, sci_padded, temp_labels_padded, times_padded, future_times_vec, true_labels_padded, lengths


def get_train_valid_loader(
        dataset,
        batch_size,
        random_seed,
        valid_size=0.2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=my_collate,
        patient_ids=None,
        labels=None
):
    """
    Split train/validation and build DataLoaders.

    Prefer StratifiedGroupKFold:
      - Group by patient_ids so the same patient never appears in both sets
      - Stratify by labels to preserve class balance
    Fall back to a simple random split if patient IDs / labels are unavailable.
    """
    num_train = len(dataset)
    indices = np.arange(num_train)

    if patient_ids is not None and labels is not None:
        # n_splits ≈ 1/valid_size, e.g. valid_size=0.2 -> 5 folds; use fold 0
        n_splits = int(1.0 / valid_size) if valid_size > 0 else 5
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)

        train_idx_list, valid_idx_list = list(sgkf.split(indices, labels, patient_ids))[0]
        train_idx = indices[train_idx_list]
        valid_idx = indices[valid_idx_list]

        print(f'Patient-level stratified split: {len(train_idx)} train samples, {len(valid_idx)} valid samples')
        print(f'Unique patients in train: {len(np.unique(patient_ids[train_idx]))}')
        print(f'Unique patients in valid: {len(np.unique(patient_ids[valid_idx]))}')
    else:
        # Simple random split
        split = int(np.floor(valid_size * num_train))
        if shuffle:
            np.random.seed(random_seed)
            shuffled_indices = indices.copy()
            np.random.shuffle(shuffled_indices)
            train_idx = shuffled_indices[split:]
            valid_idx = shuffled_indices[:split]
        else:
            train_idx = indices[split:]
            valid_idx = indices[:split]
        print(f'Simple random split: {len(train_idx)} train samples, {len(valid_idx)} valid samples')

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return (train_loader, valid_loader, len(train_idx), len(valid_idx))


# =====================================================================
#                         Hyperparameters and setup
# =====================================================================

# task: "rr" = recurrence-recovery (binary); "severity" = ordinal severity (3 classes)
task = "rr"
batch_size = 16
epochs = 500
lr = 1e-3
dropout = 0.2
num_classes = 2 if task == "rr" else 3

# Build Dataset
train_dataset = PatientTimeSeriesDataset(
    ife_images, sci_images, temporal_labels, true_labels,
    visit_times=visit_times, transform=transform_batch
)

# Labels for stratified split: if true_label is a sequence, use the first future step
if true_labels.ndim == 1:
    stratify_labels = true_labels
else:
    stratify_labels = true_labels[:, 0] if true_labels.ndim == 2 else true_labels.flatten()

train_loader, valid_loader, num_train, num_valid = get_train_valid_loader(
    train_dataset, batch_size, seed, valid_size=0.2, collate_fn=my_collate,
    patient_ids=patient_ids, labels=stratify_labels
)

# Device, model, loss, optimizer, LR scheduler
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
model = PatientTimeSeriesModel(feature_dim=64, num_classes=num_classes, task=task, dropout=dropout)
model.to(device)
# Model outputs probabilities, so use NLLLoss(log(p)); add 1e-8 to avoid log(0)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# Reduce LR by default factor 0.1 when val loss does not improve for `patience` epochs
scheduler = ReduceLROnPlateau(optimizer, "min", patience=20)


# =====================================================================
#                         Training / validation loop
# =====================================================================

best_valid_acc = 0.0
for epoch in range(epochs):
    # -------------------- Training --------------------
    model.train()
    total_loss = 0.0
    with tqdm(total=num_train, desc=f"Epoch {epoch+1}/{epochs} [Train]") as pbar:
        for batch in train_loader:
            imgs224, imgs35, labels, times, future_times, future_labels, lengths = batch
            # Move tensors to device
            imgs224 = imgs224.to(device)
            imgs35 = imgs35.to(device)
            labels = labels.to(device)
            times = times.to(device)
            future_times = future_times.to(device)
            future_labels = future_labels.to(device)
            lengths = lengths.to(device)

            # Forward: preds shape (T_fut, B, num_classes)
            # lengths selects the last valid (non-padded) T-LSTM hidden state per patient
            preds = model(imgs224, imgs35, labels, times, future_times, lengths)

            # Flatten to (B*T_fut, num_classes) and (B*T_fut,) for classification loss
            T_fut = preds.size(0)
            B = preds.size(1)
            preds_flat = preds.permute(1, 0, 2).reshape(B * T_fut, -1)
            targets_flat = future_labels.reshape(-1)

            # NLLLoss needs log-probabilities; model outputs class probabilities
            loss = criterion(torch.log(preds_flat + 1e-8), targets_flat)

            # Backward and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.update(B)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)

    # -------------------- Validation --------------------
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # no gradients during validation
        for batch in valid_loader:
            imgs224, imgs35, labels, times, future_times, future_labels, lengths = batch
            imgs224 = imgs224.to(device)
            imgs35 = imgs35.to(device)
            labels = labels.to(device)
            times = times.to(device)
            future_times = future_times.to(device)
            future_labels = future_labels.to(device)
            lengths = lengths.to(device)

            preds = model(imgs224, imgs35, labels, times, future_times, lengths)

            T_fut = preds.size(0)
            B = preds.size(1)
            preds_flat = preds.permute(1, 0, 2).reshape(B * T_fut, -1)
            targets_flat = future_labels.reshape(-1)

            loss = criterion(torch.log(preds_flat + 1e-8), targets_flat)
            valid_loss += loss.item()

            # Argmax over class dimension
            _, predicted = torch.max(preds_flat, 1)
            total += targets_flat.size(0)
            correct += (predicted == targets_flat).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets_flat.cpu().numpy())

    avg_valid_loss = valid_loss / len(valid_loader)
    valid_acc = 100.0 * correct / total

    # Macro average: equal weight per class (better under class imbalance)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {avg_loss:.4f}")
    print(f"  Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")
    print(f"  Valid Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Adjust learning rate based on validation loss
    scheduler.step(avg_valid_loss)

    # Save the best model by validation accuracy
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        print(f"*******New Best Valid Accuracy: {best_valid_acc:.2f}%*******")
        # Also store task so the evaluation script can read it automatically
        torch.save({'model_state_dict': model.state_dict(), 'task': task}, '.../Trans_TSMonitor_best.pt')
        print("Model saved!")

    # Free cached GPU memory fragments
    torch.cuda.empty_cache()
