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
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
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

loaded_train = np.load(".../TrainData_NP_Multi.npz")
train_csd = loaded_train['csd_arr']
train_label = loaded_train['labels_np_arr']

train_csd = np.array(train_csd)
train_label = np.array(train_label)

patient_ids = None
try:
    excel_path = "all_sample_information.xlsx"
    df_info = pd.read_excel(excel_path, sheet_name=0)
    df_d1_d4 = df_info[df_info['Fold'].isin(['D1', 'D2', 'D3', 'D4'])]
    patient_ids_loaded = df_d1_d4['PaID'].values
    if len(patient_ids_loaded) == len(train_label):
        patient_ids = patient_ids_loaded
        print(f'Loaded {len(patient_ids)} patient IDs matching data length')
    else:
        print(f'Warning: Patient IDs length ({len(patient_ids_loaded)}) does not match data length ({len(train_label)})')
        print('Using index-based patient IDs (each sample is a separate patient)')
        patient_ids = np.arange(len(train_label))
except Exception as e:
    print(f'Warning: Could not load patient IDs: {e}')
    print('Using index-based patient IDs (each sample is a separate patient)')
    patient_ids = np.arange(len(train_label))

transform_batch = transforms.Compose([transforms.ToTensor(),])
train_csd = [transform_batch(csd).type(torch.FloatTensor) for csd in train_csd]

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
        collate_fn=my_collate,
        patient_ids=None,
        labels=None
):
    """
    Get train and valid loaders with patient-level and stratified sampling.
    
    Args:
        dataset: The dataset
        batch_size: Batch size
        random_seed: Random seed
        valid_size: Validation set size ratio
        shuffle: Whether to shuffle
        num_workers: Number of workers
        pin_memory: Whether to pin memory
        collate_fn: Collate function
        patient_ids: Array of patient IDs for each sample (for patient-level splitting)
        labels: Array of labels for stratified sampling
    """
    num_train = len(dataset)
    indices = np.arange(num_train)

    if patient_ids is not None and labels is not None:
        n_splits = int(1.0 / valid_size) if valid_size > 0 else 10
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

train_dataset = CustomizedDataset(train_csd, train_label)
train_loader, valid_loader, num_train, num_valid = get_train_valid_loader(
    train_dataset, 64, 666, valid_size=0.2, collate_fn=my_collate,
    patient_ids=patient_ids, labels=train_label
)

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

criterion = nn.CrossEntropyLoss()
repeat = 1
r_accs = np.zeros((repeat, 6))

for r in range(repeat):
    print("repeating #" + str(r))
    model = Net()
    for param in model.parameters():
        param.requires_grad = True
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=20)

    losses_lst = []
    accs_lst = []
    perc_best = 0
    for epoch in range(0, 60):
        model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        with tqdm(total=num_train) as pbar:
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)

                # initialize location vector and hidden state
                batch_size = x.shape[0]
                output = model(x)
                predicted = torch.max(output, 1)[1]
                loss = criterion(output, y)

                # compute accuracy
                correct = torch.sum((predicted == y).float())
                acc = 100 * (correct / batch_size)

                # store
                losses.update(loss.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])

                loss.backward()
                optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(("{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc - tic), losses.avg, accs.avg)))
                pbar.update(batch_size)
                torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            losses = AverageMeter()
            accs = AverageMeter()

            best_valid_acc = 0
            for i, (x, y) in enumerate(valid_loader):
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)

                batch_size = x.shape[0]
                output = model(x)
                _, predicted = torch.max(output, 1)
                loss = criterion(output, y)

                # compute accuracy
                correct = torch.sum((predicted == y).float())
                acc = 100 * (correct / batch_size)

                # store
                losses.update(loss.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])

            losses_lst.append(losses.avg)
            accs_lst.append(accs.avg)

            if accs.avg > best_valid_acc:
                best_valid_acc_ = accs.avg
            print("epoch: " + str(epoch))
            print("validation losses avg. " + str(losses.avg))
            print("validation accs avg. " + str(accs.avg))


            torch.cuda.empty_cache()

        scheduler.step(-accs.avg)

        # Evaluate on validation set for model saving
        with torch.no_grad():
            correct = 0

            start = time.time()
            pred_label = []; true_label = []
            model.eval()
            for i, (x, y) in enumerate(valid_loader):
                x, y = x.to(device), y.to(device)

                batch_size = x.shape[0]
                output = model(x)
                _, predicted = torch.max(output, 1)
                predicted = predicted.long()
                pred_label.append(predicted.cpu().detach().numpy().tolist())
                true_label.append(y.long().cpu().detach().numpy().tolist())

                correct += torch.sum((predicted == y).float())

            end = time.time()

            perc = (100.0 * correct) / (num_valid)
            error = 100 - perc

            print("[*] Valid Acc: {}/{} ({:.6f}% - {:.6f}%)".format(correct, num_valid, perc, error))

            true_label = [element for sublist in true_label for element in sublist]
            pred_label = [element for sublist in pred_label for element in sublist]
            if perc > perc_best:
                perc_best = perc
                Preci = precision_score(true_label, pred_label, average='macro')
                Recal = recall_score(true_label, pred_label, average='macro')
                F1score = f1_score(true_label, pred_label, average='macro')
                print('*******Now Best Performance!!!!*******')
                print(
                    'Valid Accuracy {:.4f} | Valid F1-score {:.4f} | Valid Recall {:.4f} | Valid Presision {:.4f}'.format(
                        perc, F1score, Recal, Preci))
                cm = confusion_matrix(true_label, pred_label, labels=[0, 1])
                print('Confusion Matrix of Positive-Negative Classification:', cm)
                torch.save({'CSDmodel_state_dict': model.state_dict()},
                           '.../Net_SCI_0.pt')

