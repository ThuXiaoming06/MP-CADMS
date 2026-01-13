import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, auc, roc_curve
import os
import cv2
from skimage import color
import torchvision.models as models
import pandas as pd

seed=666
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

loaded_test = np.load(".../ExternalData_NP_Multi.npz")
test_csd = loaded_test['csd_arr']
test_label = loaded_test['labels_np_arr']

test_csd = np.array(test_csd)
test_label = np.array(test_label)

transform_batch = transforms.Compose([transforms.ToTensor(),])
test_csd = [transform_batch(csd).type(torch.FloatTensor) for csd in test_csd]

class CustomizedDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.Y)


def my_collate(batch):
    data = torch.cat([item[0].unsqueeze(0) for item in batch], axis=0)
    target = torch.Tensor([item[1] for item in batch])
    target = target.long()
    return [data, target]

def get_train_valid_loader(
        dataset,
        batch_size,
        random_seed,
        valid_size=0.1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=my_collate
):
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=my_collate,
    )

    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=my_collate,
    )

    return (train_loader, valid_loader, num_train - split)

test_dataset = CustomizedDataset(test_csd, test_label)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True, collate_fn=my_collate)

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

kernel_size = 2
init_channels = 12 # initial number of filters
image_channels = 1
latent_dim = 24 # latent dimension for sampling


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.CSD_model = models.resnet18()
        num_ftrs_CSD = self.CSD_model.fc.in_features
        self.CSD_model.fc = nn.Linear(num_ftrs_CSD, 2)

    def forward(self, x):
        output = self.CSD_model(x)
        return output

checkpoint_net = torch.load('.../Net_SCI_0.pt')
criterion = nn.CrossEntropyLoss()

model = Net()
model.load_state_dict(checkpoint_net['CSDmodel_state_dict'])

optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
scheduler = ReduceLROnPlateau(optimizer, "min", patience=20)

with torch.no_grad():
    num_test = len(test_dataset)
    correct = 0

    start = time.time()
    pred_label = []; true_label = []; prob_out = []
    # model.eval()
    for i, (x, y) in enumerate(test_loader):
        model.to(device)
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        # initialize location vector and hidden state
        batch_size = x.shape[0]
        output = model(x)
        _, predicted = torch.max(output, 1)
        predicted = predicted.long()
        pred_label.append(predicted.cpu().detach().numpy().tolist())
        true_label.append(y.long().cpu().detach().numpy().tolist())
        prob_out.append(output.cpu().detach().numpy().tolist())

        correct += torch.sum((predicted == y).float())

    end = time.time()

    perc = (100.0 * correct) / (num_test)
    error = 100 - perc

    print("[*] Test Acc: {}/{} ({:.4f}% - {:.4f}%)".format(correct, num_test, perc, error))

    true_label = [element for sublist in true_label for element in sublist]
    pred_label = [element for sublist in pred_label for element in sublist]
    prob_out = [element for sublist in prob_out for element in sublist]

    Preci_np = precision_score(true_label, pred_label, average='macro')
    Recal_np = recall_score(true_label, pred_label, average='macro')
    F1score_np = f1_score(true_label, pred_label, average='macro')
    print('Test Accuracy {:.4f} | Test F1-score {:.4f} | Test Recall {:.4f} | Test Presision {:.4f}'.format(
        perc, F1score_np, Recal_np, Preci_np))