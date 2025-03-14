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

train_pos_data = []
loaded1 = np.load("...")
train_pos_data.append(loaded1['imgs_arr'])
loaded2 = np.load("...")
train_pos_data.append(loaded2['imgs_arr'])
loaded3 = np.load("...")
train_pos_data.append(loaded3['imgs_arr'])
loaded4 = np.load("...")
train_pos_data.append(loaded4['imgs_arr'])
train_pos_data = np.concatenate(train_pos_data, axis=0)

loaded_test = np.load("...")
test_pos_data = loaded_test['imgs_arr']

train_neg_data = []
loaded1 = np.load("...z")
train_neg_data.append(loaded1['imgs_arr'])
loaded2 = np.load("...")
train_neg_data.append(loaded2['imgs_arr'])
loaded3 = np.load("...")
train_neg_data.append(loaded3['imgs_arr'])
loaded4 = np.load("...")
train_neg_data.append(loaded4['imgs_arr'])
train_neg_data = np.concatenate(train_neg_data, axis=0)

loaded_test = np.load("...")
test_neg_data = loaded_test['imgs_arr']

train_pos_data = np.array(train_pos_data)
test_pos_data = np.array(test_pos_data)
train_neg_data = np.array(train_neg_data)
test_neg_data = np.array(test_neg_data)

train_data = np.concatenate((train_neg_data, train_pos_data), axis=0)
test_data = np.concatenate((test_neg_data, test_pos_data), axis=0)
print('train_data.shape:', train_data.shape)
print('test_data.shape:', test_data.shape)

train_label = np.zeros((train_data.shape[0], ))
train_label[train_neg_data.shape[0]:] = 1
test_label = np.zeros((test_data.shape[0], ))
test_label[test_neg_data.shape[0]:] = 1

transform_batch = transforms.Compose([transforms.ToTensor(),])
train_data = [transform_batch(img).type(torch.FloatTensor) for img in train_data]
test_data = [transform_batch(img).type(torch.FloatTensor) for img in test_data]


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

train_dataset = CustomizedDataset(train_data, train_label)
test_dataset = CustomizedDataset(test_data, test_label)
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

class ResBlock(nn.Module):
    """
    残差块
    """
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1):
        """
        :param input_channels: 输入的通道数
        :param output_channels: 输出的通道数
        :param use_1x1conv: 是否使用1x1的卷积核[默认不使用]
        :param strides: 滑动步长
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        if self.conv3:
            x = self.conv3(x)
        y = y + x
        return y

class BottleNeck(nn.Module):
    """
    Bottleneck瓶颈层
    """
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1):
        """
        :param input_channels: 输入的通道数
        :param output_channels: 输出的通道数
        :param use_1x1conv: 是否使用1x1的卷积核[默认不使用]
        :param strides: 滑动步长
        """
        super().__init__()
        mid_channels = input_channels
        if input_channels / 4 != 0:
            mid_channels = int(mid_channels / 4)
        self.conv1 = nn.Conv2d(input_channels, mid_channels, kernel_size=1, padding=0, stride=strides)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, output_channels, kernel_size=1, padding=0, stride=strides)
        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.bn3 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        if self.conv4:
            x = self.conv4(x)
        y = y + x
        return y

class MyResNet(nn.Module):
    """
    残差网络
    """
    _loss = []
    _loss4epochs = []
    model_weights = []
    conv_layers = []

    def __init__(self):
        super().__init__()
        self._loss.clear()
        self._loss4epochs.clear()
        self.bl1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                                 nn.BatchNorm2d(64), nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.bl2 = nn.Sequential(*self.res_layer_maker(64, 64, 2, first_block=True))
        self.bl3 = nn.Sequential(*self.res_layer_maker(64, 128, 2))
        self.bl4 = nn.Sequential(*self.res_layer_maker(128, 256, 2))
        self.bottleneck = BottleNeck(256, 256)
        self.bl5 = nn.Sequential(*self.res_layer_maker(256, 512, 2))
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, 64)

    def forward(self, x):
        x = self.bl1(x)
        x = self.bl2(x)
        x = self.bl3(x)
        x = self.bl4(x)
        x = self.bottleneck(x)
        x = self.bl5(x)
        x = self.linear(self.flatten(self.pool1(x)))
        return x

    @staticmethod
    def res_layer_maker(input_channels, output_channels, num_residuals,
                        first_block=False):
        """
        调用残差块，对第一个残差块进行处理（1x1卷积核），并连接重复的残差块

        :param input_channels: 输入的通道数
        :param output_channels: 输出的通道数
        :param num_residuals: 残差块重复的次数
        :param first_block: 是否是第一个残差块
        :return: 处理好的残差块
        """
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(ResBlock(input_channels, output_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(ResBlock(output_channels, output_channels))
        return blk

# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()

        # encoder
        self.resencoder = MyResNet()
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 192)
        self.fc_mu = nn.Linear(192, latent_dim)
        self.fc_log_var = nn.Linear(192, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 96)
        self.fc3 = nn.Linear(96, 9792)

        # decoder
        self.dec1 = nn.ConvTranspose2d(in_channels=96, out_channels=init_channels * 4, kernel_size=kernel_size, stride=2)
        self.dec2 = nn.ConvTranspose2d(in_channels=init_channels * 4, out_channels=init_channels * 2, kernel_size=kernel_size, stride=2)
        self.dec3 = nn.ConvTranspose2d(in_channels=init_channels * 2, out_channels=init_channels, kernel_size=kernel_size, stride=2)
        self.dec4 = nn.ConvTranspose2d(in_channels=init_channels, out_channels=image_channels, kernel_size=kernel_size, stride=2)
        self.same_size = nn.AdaptiveAvgPool2d((128, 128))

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def forward(self, x):
        # encoding
        x = F.relu(self.resencoder(x))
        batch, _ = x.shape
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        return z

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#         self.dropout2 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(9792, 512)
        self.fc2 = nn.Linear(512, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

checkpoint = torch.load('...')
criterion = nn.CrossEntropyLoss()
repeat = 1
r_accs = np.zeros((repeat, 6))

for r in range(repeat):
    print("repeating #" + str(r))
    feature_extractor = ConvVAE()
    feature_extractor.load_state_dict(checkpoint['model_state_dict'])
    for param in feature_extractor.parameters():
        param.requires_grad = True
    model = Net()
    for param in model.parameters():
        param.requires_grad = True
    model.to(device)
    model.train()
    feature_extractor.to(device)
    feature_extractor.train()
    optimizer1 = torch.optim.Adam(model.parameters(), lr=2.5e-3)
    scheduler1 = ReduceLROnPlateau(optimizer1, "min", patience=20)
    optimizer2 = torch.optim.Adam(feature_extractor.parameters(), lr=2.5e-3)
    scheduler2 = ReduceLROnPlateau(optimizer2, "min", patience=20)

    losses_lst = []
    accs_lst = []
    perc_best = 0
    for epoch in range(0, 60):
        feature_extractor.train()
        model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        model.train()
        feature_extractor.train()
        with tqdm(total=num_train) as pbar:
            for i, (x, y) in enumerate(train_loader):
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                x, y = x.to(device), y.to(device)

                # initialize location vector and hidden state
                batch_size = x.shape[0]
                output = model(feature_extractor(x))
                predicted = torch.max(output, 1)[1]
                loss = criterion(output, y)

                # compute accuracy
                correct = torch.sum((predicted == y).float())
                acc = 100 * (correct / batch_size)

                # store
                losses.update(loss.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])

                loss.backward()
                optimizer1.step()
                optimizer2.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(("{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc - tic), losses.avg, accs.avg)))
                pbar.update(batch_size)
                torch.cuda.empty_cache()

        model.eval()
        feature_extractor.eval()
        with torch.no_grad():
            losses = AverageMeter()
            accs = AverageMeter()

            best_valid_acc = 0
            for i, (x, y) in enumerate(valid_loader):
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                x, y = x.to(device), y.to(device)

                batch_size = x.shape[0]
                output = model(feature_extractor(x))
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

        scheduler1.step(-accs.avg)
        scheduler2.step(-accs.avg)
