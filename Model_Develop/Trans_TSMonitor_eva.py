import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time

try:
    from torchdiffeq import odeint
except ImportError:
    def odeint(func, x0, t):
        t = torch.tensor(t, dtype=torch.float32)
        return x0.unsqueeze(0).repeat(len(t), 1, 1)

# Model definition (same as training file)
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

# Load test data
loaded_test = np.load(".../Test_TS_RRSP.npz")
test_ife_images = loaded_test['IFE_image']
test_temporal_labels = loaded_test['temporal_label']
test_sci_images = loaded_test['SCI']
test_true_labels = loaded_test['true_label']

print(f'Loaded test data shapes:')
print(f'  IFE_image: {test_ife_images.shape}')
print(f'  temporal_label: {test_temporal_labels.shape}')
print(f'  SCI: {test_sci_images.shape}')
print(f'  true_label: {test_true_labels.shape}')

# Convert to numpy arrays if needed
test_ife_images = np.array(test_ife_images)
test_temporal_labels = np.array(test_temporal_labels)
test_sci_images = np.array(test_sci_images)
test_true_labels = np.array(test_true_labels)

transform_batch = transforms.Compose([transforms.ToTensor()])

# Test Dataset
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
        ife_seq = self.ife_images[idx]
        sci_seq = self.sci_images[idx]
        temp_labels = self.temporal_labels[idx]
        true_label = self.true_labels[idx]
        
        # Convert to torch tensors and handle different input formats
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
        
        # Convert to torch tensors
        ife_seq = torch.from_numpy(ife_seq).float()
        sci_seq = torch.from_numpy(sci_seq).float()
        temp_labels = torch.from_numpy(temp_labels).long()
        
        # Handle true_label
        if true_label.ndim == 0:
            true_label = np.array([true_label])
        true_label = torch.from_numpy(true_label).long()
        
        # Generate time points
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
    
    # Use first sample's future_times
    future_times_vec = future_times[0]
    
    return ife_padded, sci_padded, temp_labels_padded, times_padded, future_times_vec, true_labels_padded

# Prepare test dataset
test_dataset = PatientTimeSeriesDataset(test_ife_images, test_sci_images, test_temporal_labels, test_true_labels, transform=transform_batch)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=my_collate)

# Load model
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

num_classes = len(np.unique(test_true_labels)) if test_true_labels.ndim == 1 else 4
model = PatientTimeSeriesModel(feature_dim=64, num_classes=num_classes)
checkpoint = torch.load('.../Trans_TSMonitor_best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Evaluation
print("\n=== Evaluating on Test Set ===")
correct = 0
total = 0
all_preds = []
all_labels = []

start_time = time.time()
with torch.no_grad():
    for batch in test_loader:
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
        
        _, predicted = torch.max(preds_flat, 1)
        total += targets_flat.size(0)
        correct += (predicted == targets_flat).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets_flat.cpu().numpy())

end_time = time.time()

# Calculate metrics
test_acc = 100.0 * correct / total
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

