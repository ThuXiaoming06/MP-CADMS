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
train_data = loaded_train['imgs_arr']
train_csd = loaded_train['csd_arr']
train_label = loaded_train['labels_np_arr']

train_data = np.array(train_data)
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
train_data = [transform_batch(img).type(torch.FloatTensor) for img in train_data]
train_csd = [transform_batch(csd).type(torch.FloatTensor) for csd in train_csd]

class CustomizedDataset(data.Dataset):
    def __init__(self, X1, X2, Y):
        self.X1 = X1
        self.X2 = X2
        self.Y = Y

    def __getitem__(self, index):
        return self.X1[index], self.X2[index], self.Y[index]

    def __len__(self):
        return len(self.Y)


def my_collate(batch):
    data1 = torch.cat([item[0].unsqueeze(0) for item in batch], axis=0)
    data2 = torch.cat([item[1].unsqueeze(0) for item in batch], axis=0)
    target = torch.Tensor([item[2] for item in batch])
    target = target.long()
    return [data1, data2, target]

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

train_dataset = CustomizedDataset(train_data, train_csd, train_label)
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

class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#         self.dropout2 = nn.Dropout(0.1)
        self.CSD_model = models.resnet18()
        #self.CSD_model = models.vgg19()
        num_ftrs_CSD = self.CSD_model.fc.in_features
        self.CSD_model.fc = nn.Linear(num_ftrs_CSD, 512)

        self.fc1_1 = nn.Linear(9792, 1024)
        self.fc1_2 = nn.Linear(1024, 512)
        self.trans_block_multi1 = TransformerLayer(512, 4)
        self.trans_block_multi2 = TransformerLayer(512, 4)
        self.trans_block_multi3 = TransformerLayer(512, 4)

        self.fc2_1 = nn.Linear(512 * 2, 256)
        self.fc2_2 = nn.Linear(256, 2)


    def forward(self, x1, x2):
        #print(x2)
        batch, _ = x1.shape

        x_csd = self.CSD_model(x2)
        #print(x_csd)

        x_ife = self.fc1_1(x1)
        x_ife = F.relu(x_ife)
        x_ife = self.fc1_2(x_ife)

        x_multi = torch.cat([x_csd, x_ife], dim=1)
        x_multi = F.relu(x_multi).view(batch, 2, 512)
        x_multi = self.trans_block_multi1(x_multi).view(batch, -1)
        x_multi = F.relu(x_multi).view(batch, 2, 512)
        x_multi = self.trans_block_multi2(x_multi).view(batch, -1)
        x_multi = F.relu(x_multi).view(batch, 2, 512)
        x_multi = self.trans_block_multi3(x_multi).view(batch, -1)
        #print(x_multi)
        x = F.relu(self.fc2_1(x_multi))
        x = self.fc2_2(x)
        #print(x)

        return x

checkpoint = torch.load('.../Pretrain_IFEextractor.pt')
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
    optimizer1 = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    scheduler1 = ReduceLROnPlateau(optimizer1, "min", patience=20)
    optimizer2 = torch.optim.Adam(feature_extractor.parameters(), lr=2.8e-4)
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
            for i, (x1, x2, y) in enumerate(train_loader):
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)

                # initialize location vector and hidden state
                batch_size = x1.shape[0]
                output = model(feature_extractor(x1), x2)
                predicted = torch.max(output, 1)[1]
                loss = criterion(output, y)

                # compute accuracy
                correct = torch.sum((predicted == y).float())
                acc = 100 * (correct / batch_size)

                # store
                losses.update(loss.item(), x1.size()[0])
                accs.update(acc.item(), x1.size()[0])

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
            for i, (x1, x2, y) in enumerate(valid_loader):
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)

                batch_size = x1.shape[0]
                output = model(feature_extractor(x1), x2)
                _, predicted = torch.max(output, 1)
                loss = criterion(output, y)

                # compute accuracy
                correct = torch.sum((predicted == y).float())
                acc = 100 * (correct / batch_size)

                # store
                losses.update(loss.item(), x1.size()[0])
                accs.update(acc.item(), x1.size()[0])

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

        # Evaluate on validation set for model saving
        with torch.no_grad():
            correct = 0

            start = time.time()
            pred_label = []; true_label = []
            model.eval()
            feature_extractor.eval()
            for i, (x1, x2, y) in enumerate(valid_loader):
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)

                batch_size = x1.shape[0]
                output = model(feature_extractor(x1), x2)
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
                torch.save({'feature_extractor_state_dict': feature_extractor.state_dict()},
                           '.../ConVAE_multi_0.pt')
                torch.save({'IFEmodel_state_dict': model.state_dict()},
                           '.../Net_multi_0.pt')
