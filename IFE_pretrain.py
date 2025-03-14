import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color
import cv2
import torch
import time
from tqdm import tqdm
import matplotlib
from torchvision import transforms
from torchvision.utils import make_grid
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from engine import train, validate
#from utils import save_reconstructed_images, image_to_vid, save_loss_plot

matplotlib.style.use('ggplot')

root_addr = 'D:/An_PositiveLevel/Data/'
suffix_addr = 'Dataset_all/'
imgs_addr = root_addr + suffix_addr
#Tiaoma_addr = root_addr + '1stCV_pretrain.xlsx'
#Tiaoma_addr = root_addr + 'CV1_4_all.xlsx'
Tiaoma_addr = root_addr + 'CV1_2_3_IV_pretrain.xlsx'
# read the label file
labels = pd.read_excel(Tiaoma_addr, sheet_name='Sheet1', header=0)
labels = labels[['条码号']]
print(labels)

imgs_lst = []; i = 0
for img_id in list(labels['条码号']):
    i += 1
    print('The current image number is:', i)
    imgs_lst.append(plt.imread(imgs_addr + str(img_id) + '.gif'))
imgs_lst = [cv2.resize(color.rgb2gray(img[20:-25, :, :]), (128, 128), interpolation=cv2.INTER_NEAREST) for img in imgs_lst]
imgs_arr = np.array(imgs_lst)

transform_batch = transforms.Compose([transforms.ToTensor(), ])
imgs = [transform_batch(img).type(torch.FloatTensor).permute(1, 0, 2).reshape(1, 128, 128) for img in imgs_arr]

X_train, X_test = train_test_split(imgs, test_size=0.01, random_state=666)
print('X_train.shape:', len(X_train)); print('X_test.shape:', len(X_test))
print('X_train[0].shape:', X_train[0].shape)

class CustomizedDataset(data.Dataset):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return len(self.X)

def my_collate(batch):
    data = torch.cat([item.unsqueeze(0) for item in batch], axis=0)
    return [data]

def get_train_valid_loader(
    dataset,
    batch_size,
    random_seed,
    valid_size=0.01,
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

    return (train_loader, valid_loader, num_train - split, split)

train_dataset = CustomizedDataset(X_train)
test_dataset = CustomizedDataset(X_test)
train_loader, valid_loader, num_train, num_valid = get_train_valid_loader(train_dataset, 64, 666, collate_fn=my_collate)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True, collate_fn=my_collate)
print('num_valid:', num_valid)

kernel_size = 2  # (4, 4) kernel
init_channels = 12  # initial number of filters
image_channels = 1
latent_dim = 24  # latent dimension for sampling

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
        #self.resencoder = ResNet(ResidualBlock)
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
        z = z.view(-1, 96, 6, 17)

        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))

        #         reconstruction = torch.sigmoid(self.dec4(x))
        reconstruction = torch.sigmoid(self.same_size(self.dec4(x)))
        #         print(reconstruction.shape)
        return reconstruction, mu, log_var

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
model = ConvVAE().to(device)

# set the learning parameters
#lr = 0.001
lr = 2.5e-4
epochs = 200
batch_size = 32
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#criterion = nn.BCELoss(reduction='mean')
criterion = nn.MSELoss()
# a list to save all the reconstructed images in PyTorch grid format
grid_images = []

train_loss = []
valid_loss = []
best_train_loss = 1
best_valid_loss = 1

for epoch in range(epochs):
    train_epoch_loss = train(
        model, train_loader, num_train, device, optimizer, criterion
    )
    valid_epoch_loss, recon_images = validate(
        model, valid_loader, num_valid, device, criterion
    )

    if train_epoch_loss <= best_train_loss:
        best_train_loss = train_epoch_loss
        #torch.save({'model_state_dict': model.state_dict()}, 'D:/An_PositiveLevel/Model_save/pretext_model_resnet_train.pt')
    if valid_epoch_loss <= best_valid_loss:
        best_valid_loss = valid_epoch_loss
        #torch.save({'model_state_dict': model.state_dict()}, 'D:/An_PositiveLevel/Model_save/pretext_model_resnet_valid.pt')

    if epoch % 5 == 0:
        '''
        torch.save({'model_state_dict': model.state_dict()},
                   'D:/An_PositiveLevel/Model_save_CV/CV1_pretrain/CV1_pretext_train2_' + str(int(epoch)) + '.pt')
        torch.save({'model_state_dict': model.state_dict()},
                   'D:/An_PositiveLevel/Model_save_CV/CV1_pretrain/CV1_pretext_valid2_' + str(int(epoch)) + '.pt')
        '''
        #torch.save({'model_state_dict': model.state_dict()},
        #           'D:/An_PositiveLevel/Model_save_CV/CV1_2_3_IV_pretrain/CV1_2_3_IV_pretext_train_' + str(int(epoch)) + '.pt')
        #torch.save({'model_state_dict': model.state_dict()},
        #           'D:/An_PositiveLevel/Model_save_CV/CV1_2_3_IV_pretrain/CV1_2_3_IV_pretext_valid_' + str(int(epoch)) + '.pt')
        torch.save({'model_state_dict': model.state_dict()},
                   'D:/An_PositiveLevel/Model_save_CV/image_vis_test/pre_imgvis_train_' + str(int(epoch)) + '.pt')
        torch.save({'model_state_dict': model.state_dict()},
                   'D:/An_PositiveLevel/Model_save_CV/image_vis_test/pre_imgvis_valid_' + str(int(epoch)) + '.pt')

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    image_grid = make_grid(recon_images.detach().cpu())
# save the reconstructed images from the validation loop
#     save_reconstructed_images(image_grid, epoch+1)
# convert the reconstructed images to PyTorch image grid format
    grid_images.append(image_grid)
    print('Iter {:04d} | Train Loss {:.6f} | Valid Loss {:.6f}| Best Train Loss {:.6f} |  Best Valid Loss {:.6f}'.format(epoch, train_epoch_loss, valid_epoch_loss, best_train_loss, best_valid_loss))
#print(f"Train Loss: {train_epoch_loss:.4f}")
#print(f"Val Loss: {valid_epoch_loss:.4f}")
#print(f"Epoch {epoch + 1} of {epochs}")