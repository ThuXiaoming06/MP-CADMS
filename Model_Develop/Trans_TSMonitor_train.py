import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
import time

# Fallback for odeint if torchdiffeq is not installed
try:
    from torchdiffeq import odeint
except ImportError:
    def odeint(func, x0, t):
        t = torch.tensor(t, dtype=torch.float32)
        return x0.unsqueeze(0).repeat(len(t), 1, 1)

seed = 666
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Model definition (same as before)
class CNNFeatureExtractor(nn.Module):
    def __init__(self, pretrained=False, output_dim=64):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)
        modules = list(backbone.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(backbone.fc.in_features, output_dim)
    def forward(self, x):
        feat = self.encoder(x).squeeze(-1).squeeze(-1)
        return self.fc(feat)

class LabelMLP(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, labels):
        return self.mlp(labels.unsqueeze(-1).float())

class FusionTransformer(nn.Module):
    def __init__(self, embed_dim=64, n_heads=4, num_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
    def forward(self, a, b):
        seq = torch.stack([a, b], dim=0)
        return self.transformer(seq)[0]

class TimeAwareTransformer(nn.Module):
    def __init__(self, embed_dim=64, n_heads=4, num_layers=2):
        super().__init__()
        self.time_embed = nn.Linear(1, embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
    def forward(self, feats, times):
        t_emb = self.time_embed(times.unsqueeze(-1))
        return self.transformer(feats + t_emb)

class ODEFunc(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    def forward(self, t, x):
        return self.net(x)

class PatientTimeSeriesModel(nn.Module):
    def __init__(self, feature_dim=64, num_classes=4):
        super().__init__()
        self.cnn_large = CNNFeatureExtractor(output_dim=feature_dim)
        self.cnn_small = CNNFeatureExtractor(output_dim=feature_dim)
        self.label_mlp = LabelMLP(output_dim=feature_dim)
        self.fuse_img = FusionTransformer(embed_dim=feature_dim)
        self.fuse_all = FusionTransformer(embed_dim=feature_dim)
        self.seq_fuse = TimeAwareTransformer(embed_dim=feature_dim)
        self.odefunc = ODEFunc(dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, imgs_large, imgs_small, labels, times, future_times):
        B, T, C, H, W = imgs_large.size()
        feats = []
        for t in range(T):
            v1 = self.cnn_large(imgs_large[:, t])
            v2 = self.cnn_small(imgs_small[:, t])
            v3 = self.label_mlp(labels[:, t])
            v4 = self.fuse_img(v1, v2)
            v5 = self.fuse_all(v4, v3)
            feats.append(v5)
        seq_feats = torch.stack(feats, dim=0)
        seq_times = times.transpose(0, 1)
        hist = self.seq_fuse(seq_feats, seq_times)
        x0 = hist[-1]
        pred_states = odeint(self.odefunc, x0, future_times)
        logits = self.classifier(pred_states)
        return torch.softmax(logits, dim=-1)

# Load training data
loaded_train = np.load(".../Train_TS_RRSP.npz")
ife_images = loaded_train['IFE_image']  # Shape: (N, T, H, W, C) or similar
temporal_labels = loaded_train['temporal_label']  # Shape: (N, T)
sci_images = loaded_train['SCI']  # Shape: (N, T, H, W, C) or similar
true_labels = loaded_train['true_label']  # Shape: (N, T_future) or (N,)

print(f'Loaded data shapes:')
print(f'  IFE_image: {ife_images.shape}')
print(f'  temporal_label: {temporal_labels.shape}')
print(f'  SCI: {sci_images.shape}')
print(f'  true_label: {true_labels.shape}')

# Convert to numpy arrays if needed
ife_images = np.array(ife_images)
temporal_labels = np.array(temporal_labels)
sci_images = np.array(sci_images)
true_labels = np.array(true_labels)

# Transform images
transform_batch = transforms.Compose([transforms.ToTensor()])

# Load patient IDs for patient-level splitting
patient_ids = None
try:
    excel_path = "all_sample_information.xlsx"
    df_info = pd.read_excel(excel_path, sheet_name=0)
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

# Patient Time Series Dataset
class PatientTimeSeriesDataset(Dataset):
    def __init__(self, ife_images, sci_images, temporal_labels, true_labels, transform=None):
        self.ife_images = ife_images
        self.sci_images = sci_images
        self.temporal_labels = temporal_labels
        self.true_labels = true_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.true_labels)
    
    def __getitem__(self, idx):
        # Get data for this patient
        ife_seq = self.ife_images[idx]  # (T, H, W, C) or (T, C, H, W)
        sci_seq = self.sci_images[idx]  # (T, H, W, C) or (T, C, H, W)
        temp_labels = self.temporal_labels[idx]  # (T,)
        true_label = self.true_labels[idx]  # (T_future,) or scalar
        
        # Convert to torch tensors and handle different input formats
        if ife_seq.ndim == 4:
            # Assume (T, H, W, C) -> convert to (T, C, H, W)
            if ife_seq.shape[-1] == 3 or ife_seq.shape[-1] == 1:
                ife_seq = np.transpose(ife_seq, (0, 3, 1, 2))
            # If already (T, C, H, W), keep as is
        elif ife_seq.ndim == 3:
            # Assume (T, H, W) -> add channel dimension
            ife_seq = ife_seq[:, np.newaxis, :, :]
        
        if sci_seq.ndim == 4:
            if sci_seq.shape[-1] == 3 or sci_seq.shape[-1] == 1:
                sci_seq = np.transpose(sci_seq, (0, 3, 1, 2))
        elif sci_seq.ndim == 3:
            sci_seq = sci_seq[:, np.newaxis, :, :]
        
        # Convert to torch tensors
        ife_seq = torch.from_numpy(ife_seq).float()
        sci_seq = torch.from_numpy(sci_seq).float()
        temp_labels = torch.from_numpy(temp_labels).long()
        
        # Handle true_label - if scalar, convert to array
        if true_label.ndim == 0:
            true_label = np.array([true_label])
        true_label = torch.from_numpy(true_label).long()
        
        # Generate time points (assuming sequential)
        T_hist = len(temp_labels)
        T_future = len(true_label)
        times = torch.arange(0, T_hist, dtype=torch.float32)
        future_times = torch.arange(T_hist, T_hist + T_future, dtype=torch.float32)
        
        return ife_seq, sci_seq, temp_labels, times, future_times, true_label

def my_collate(batch):
    """Custom collate function for variable-length sequences"""
    ife_seqs, sci_seqs, temp_labels, times, future_times, true_labels = zip(*batch)
    
    # Find max sequence length
    max_T = max(len(tl) for tl in temp_labels)
    max_T_fut = max(len(fl) for fl in true_labels)
    
    B = len(batch)
    # Get image dimensions from first sample
    _, C, H, W = ife_seqs[0].shape
    
    # Pad sequences to same length
    ife_padded = torch.zeros(B, max_T, C, H, W)
    sci_padded = torch.zeros(B, max_T, C, H, W)
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
    
    # Use first sample's future_times (assuming all have same future times)
    future_times_vec = future_times[0]
    
    return ife_padded, sci_padded, temp_labels_padded, times_padded, future_times_vec, true_labels_padded

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
    Get train and valid loaders with patient-level and stratified sampling.
    """
    num_train = len(dataset)
    indices = np.arange(num_train)

    if patient_ids is not None and labels is not None:
        n_splits = int(1.0 / valid_size) if valid_size > 0 else 5
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)

        train_idx_list, valid_idx_list = list(sgkf.split(indices, labels, patient_ids))[0]
        train_idx = indices[train_idx_list]
        valid_idx = indices[valid_idx_list]

        print(f'Patient-level stratified split: {len(train_idx)} train samples, {len(valid_idx)} valid samples')
        print(f'Unique patients in train: {len(np.unique(patient_ids[train_idx]))}')
        print(f'Unique patients in valid: {len(np.unique(patient_ids[valid_idx]))}')
    else:
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

# Hyperparameters
batch_size = 2
epochs = 60
lr = 1e-3
num_classes = len(np.unique(true_labels)) if true_labels.ndim == 1 else 4

# Prepare dataset
train_dataset = PatientTimeSeriesDataset(ife_images, sci_images, temporal_labels, true_labels, transform=transform_batch)

# Get labels for stratification (use true_label, if scalar use as is, if array use first element)
if true_labels.ndim == 1:
    stratify_labels = true_labels
else:
    stratify_labels = true_labels[:, 0] if true_labels.ndim == 2 else true_labels.flatten()

train_loader, valid_loader, num_train, num_valid = get_train_valid_loader(
    train_dataset, batch_size, seed, valid_size=0.2, collate_fn=my_collate,
    patient_ids=patient_ids, labels=stratify_labels
)

# Model, loss, optimizer
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
model = PatientTimeSeriesModel(feature_dim=64, num_classes=num_classes)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, "min", patience=10)

# Training loop
best_valid_acc = 0.0
for epoch in range(epochs):
    # Training phase
    model.train()
    total_loss = 0.0
    with tqdm(total=num_train, desc=f"Epoch {epoch+1}/{epochs} [Train]") as pbar:
        for batch in train_loader:
            imgs224, imgs35, labels, times, future_times, future_labels = batch
            imgs224 = imgs224.to(device)
            imgs35 = imgs35.to(device)
            labels = labels.to(device)
            times = times.to(device)
            future_times = future_times.to(device)
            future_labels = future_labels.to(device)

            # Forward
            preds = model(imgs224, imgs35, labels, times, future_times)

            # preds: (T_fut, B, num_classes), future_labels: (B, T_fut)
            T_fut = preds.size(0)
            B = preds.size(1)
            preds_flat = preds.permute(1,0,2).reshape(B*T_fut, -1)
            targets_flat = future_labels.reshape(-1)

            loss = criterion(preds_flat, targets_flat)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.update(B)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in valid_loader:
            imgs224, imgs35, labels, times, future_times, future_labels = batch
            imgs224 = imgs224.to(device)
            imgs35 = imgs35.to(device)
            labels = labels.to(device)
            times = times.to(device)
            future_times = future_times.to(device)
            future_labels = future_labels.to(device)

            preds = model(imgs224, imgs35, labels, times, future_times)
            
            T_fut = preds.size(0)
            B = preds.size(1)
            preds_flat = preds.permute(1,0,2).reshape(B*T_fut, -1)
            targets_flat = future_labels.reshape(-1)
            
            loss = criterion(preds_flat, targets_flat)
            valid_loss += loss.item()
            
            _, predicted = torch.max(preds_flat, 1)
            total += targets_flat.size(0)
            correct += (predicted == targets_flat).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets_flat.cpu().numpy())
    
    avg_valid_loss = valid_loss / len(valid_loader)
    valid_acc = 100.0 * correct / total
    
    # Calculate metrics
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {avg_loss:.4f}")
    print(f"  Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")
    print(f"  Valid Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    scheduler.step(avg_valid_loss)
    
    # Save best model
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        print(f"*******New Best Valid Accuracy: {best_valid_acc:.2f}%*******")
        torch.save({'model_state_dict': model.state_dict()}, '.../Trans_TSMonitor_best.pt')
        print("Model saved!")
    
    torch.cuda.empty_cache()

