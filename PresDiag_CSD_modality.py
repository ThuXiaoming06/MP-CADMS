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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import os
import cv2
from skimage import color
import torchvision.models as models

seed=666
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_pos_csd = []
loaded1 = np.load("...")
train_pos_csd.append(loaded1['csd_arr'])
loaded2 = np.load("...")
train_pos_csd.append(loaded2['csd_arr'])
loaded3 = np.load("...")
train_pos_csd.append(loaded3['csd_arr'])
loaded4 = np.load("...")
train_pos_csd.append(loaded4['csd_arr'])
train_pos_csd = np.concatenate(train_pos_csd, axis=0)

loaded_test = np.load("D...")
test_pos_csd = loaded_test['csd_arr']

train_neg_csd = []
loaded1 = np.load("...")
train_neg_csd.append(loaded1['csd_arr'])
loaded2 = np.load("...")
train_neg_csd.append(loaded2['csd_arr'])
loaded3 = np.load("...")
train_neg_csd.append(loaded3['csd_arr'])
loaded4 = np.load("...")
train_neg_csd.append(loaded4['csd_arr'])
train_neg_csd = np.concatenate(train_neg_csd, axis=0)

loaded_test = np.load("...")
test_neg_csd = loaded_test['csd_arr']

train_pos_csd = np.array(train_pos_csd)
test_pos_csd = np.array(test_pos_csd)
train_neg_csd = np.array(train_neg_csd)
test_neg_csd = np.array(test_neg_csd)

train_csd = np.concatenate((train_neg_csd, train_pos_csd), axis=0)
test_csd = np.concatenate((test_neg_csd, test_pos_csd), axis=0)
print('train_csd.shape:', train_csd.shape)
print('test_csd.shape:', test_csd.shape)

train_label = np.zeros((train_csd.shape[0], ))
train_label[train_neg_csd.shape[0]:] = 1
test_label = np.zeros((test_csd.shape[0], ))
test_label[test_neg_csd.shape[0]:] = 1

transform_batch = transforms.Compose([transforms.ToTensor(),])
train_csd = [transform_batch(csd).type(torch.FloatTensor) for csd in train_csd]
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

train_dataset = CustomizedDataset(train_csd, train_label)
test_dataset = CustomizedDataset(test_csd, test_label)
train_loader, valid_loader, num_train = get_train_valid_loader(train_dataset, 64, 66, collate_fn=my_collate)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-3)
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

