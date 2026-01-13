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
import torchvision.models as models
import pandas as pd

seed=666
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_csd = []; y_pl_train = []; y_pi_train = []
CV_xuhao = np.array([1, 2, 3, 4])
for i in CV_xuhao:
    loaded = np.load(".../D" + str(i) + ".npz")
    train_csd.append(loaded['csd_arr'])
    y_pl_train.append(loaded['labels_pl_arr'])
    y_pi_train.append(loaded['labels_pi_arr'])

train_csd = np.concatenate(train_csd, axis=0)
print('train_csd.shape:', train_csd.shape)
y_pl_train = np.concatenate(y_pl_train, axis=0)
y_pi_train = np.concatenate(y_pi_train, axis=0)

patient_ids = None
try:
    excel_path = "all_sample_information.xlsx"
    df_info = pd.read_excel(excel_path, sheet_name=0)
    df_d1_d4 = df_info[df_info['Fold'].isin(['D1', 'D2', 'D3', 'D4'])]
    patient_ids_loaded = df_d1_d4['PaID'].values
    if len(patient_ids_loaded) == len(y_pl_train):
        patient_ids = patient_ids_loaded
        print(f'Loaded {len(patient_ids)} patient IDs matching data length')
    else:
        print(f'Warning: Patient IDs length ({len(patient_ids_loaded)}) does not match data length ({len(y_pl_train)})')
        print('Using index-based patient IDs (each sample is a separate patient)')
        patient_ids = np.arange(len(y_pl_train))
except Exception as e:
    print(f'Warning: Could not load patient IDs: {e}')
    print('Using index-based patient IDs (each sample is a separate patient)')
    patient_ids = np.arange(len(y_pl_train))

transform_batch = transforms.Compose([transforms.ToTensor(), ])
train_csd = [transform_batch(csd).type(torch.FloatTensor) for csd in train_csd]

class CustomizedDataset(data.Dataset):
    def __init__(self, X, Y_pl, Y_pi):
        self.X = X
        self.Y_pl = Y_pl
        self.Y_pi = Y_pi

    def __getitem__(self, index):
        return self.X[index], self.Y_pl[index], self.Y_pi[index]

    def __len__(self):
        return len(self.Y_pl)


def my_collate(batch):
    data1 = torch.cat([item[0].unsqueeze(0) for item in batch], axis=0)
    target_pl = torch.Tensor([item[1] for item in batch])
    target_pl = target_pl.long()
    target_pi = torch.Tensor([item[2] for item in batch])
    target_pi = target_pi.long()
    return [data1, target_pl, target_pi]


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
        labels: Array of labels for stratified sampling (e.g., y_pl_train or y_pi_train)
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

train_dataset = CustomizedDataset(train_csd, y_pl_train, y_pi_train)
train_loader, valid_loader, num_train, num_valid = get_train_valid_loader(
    train_dataset, 64, 666, valid_size=0.2, collate_fn=my_collate,
    patient_ids=patient_ids, labels=y_pl_train)

kernel_size = 2
init_channels = 12 # initial number of filters
image_channels = 1
latent_dim = 24 # latent dimension for sampling

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
        num_ftrs_CSD = self.CSD_model.fc.in_features
        self.CSD_model.fc = nn.Linear(num_ftrs_CSD, 512)

        self.fc2 = nn.Linear(512, 256)
        self.fc3_pl_wp = nn.Linear(256, 128)
        self.fc3_pl_p = nn.Linear(256, 128)
        self.fc3_pl_sp = nn.Linear(256, 128)
        self.fcc3_pl = nn.Linear(128*3, 128)
        self.fc3_pl_weight = nn.Linear(128, 3)
        self.fc4_pl = nn.Linear(128 * 2, 2)
        #self.fc3_pl_weight = nn.Linear(128*3, 3)
        #self.fc4_pl = nn.Linear(128*4, 2)

        self.fc3_pi_Ak = nn.Linear(256, 128)
        self.fc3_pi_AL = nn.Linear(256, 128)
        self.fc3_pi_Gk = nn.Linear(256, 128)
        self.fc3_pi_GL = nn.Linear(256, 128)
        self.fc3_pi_Mk = nn.Linear(256, 128)
        self.fc3_pi_ML = nn.Linear(256, 128)
        self.fc3_pi_k = nn.Linear(256, 128)
        self.fc3_pi_L = nn.Linear(256, 128)
        self.fcc3_pi = nn.Linear(128 * 8, 128)
        self.fc3_pi_weight = nn.Linear(128, 8)
        self.fc4_pi = nn.Linear(128 * 2, 8)
        #self.fc3_pi_weight = nn.Linear(128 * 8, 8)
        #self.fc4_pi = nn.Linear(128*9, 8)

        self.trans_block_pi = TransformerLayer(128, 4)
        self.trans_block_pl = TransformerLayer(128, 4)

    def forward(self, x):
        #print(x.shape)
        batch, _, _, _ = x.shape

        x = self.CSD_model(x)
        x = self.fc2(x)
        x = F.relu(x)

        #positive isotype classification task
        x_pi_mid_Ak = self.fc3_pi_Ak(x); x_pi_mid_AL = self.fc3_pi_AL(x)
        x_pi_mid_Gk = self.fc3_pi_Gk(x); x_pi_mid_GL = self.fc3_pi_GL(x)
        x_pi_mid_Mk = self.fc3_pi_Mk(x); x_pi_mid_ML = self.fc3_pi_ML(x)
        x_pi_mid_k = self.fc3_pi_k(x); x_pi_mid_L = self.fc3_pi_L(x)
        x_pi_mid = torch.cat([x_pi_mid_Ak, x_pi_mid_AL, x_pi_mid_Gk, x_pi_mid_GL, x_pi_mid_Mk, x_pi_mid_ML,
                              x_pi_mid_k, x_pi_mid_L], dim=1)
        x_pi_mid = F.relu(self.fcc3_pi(x_pi_mid))
        x_pi_weight = F.softmax(self.fc3_pi_weight(x_pi_mid), dim=1)

        #x_pi_weight = F.softmax(self.fc3_pi_weight(x_pi_mid), dim=1)
        xx_Ak = x_pi_weight[:, 0].view(x_pi_weight.shape[0], 1).repeat(1, x_pi_mid_Ak.shape[1])
        xx_AL = x_pi_weight[:, 1].view(x_pi_weight.shape[0], 1).repeat(1, x_pi_mid_AL.shape[1])
        xx_Gk = x_pi_weight[:, 2].view(x_pi_weight.shape[0], 1).repeat(1, x_pi_mid_Gk.shape[1])
        xx_GL = x_pi_weight[:, 3].view(x_pi_weight.shape[0], 1).repeat(1, x_pi_mid_GL.shape[1])
        xx_Mk = x_pi_weight[:, 4].view(x_pi_weight.shape[0], 1).repeat(1, x_pi_mid_Mk.shape[1])
        xx_ML = x_pi_weight[:, 5].view(x_pi_weight.shape[0], 1).repeat(1, x_pi_mid_ML.shape[1])
        xx_k = x_pi_weight[:, 6].view(x_pi_weight.shape[0], 1).repeat(1, x_pi_mid_k.shape[1])
        xx_L = x_pi_weight[:, 7].view(x_pi_weight.shape[0], 1).repeat(1, x_pi_mid_L.shape[1])

        x_pi_mid_fi = x_pi_mid_Ak * xx_Ak + x_pi_mid_AL * xx_AL + x_pi_mid_Gk * xx_Gk + x_pi_mid_GL * xx_GL + \
             x_pi_mid_Mk * xx_Mk + x_pi_mid_ML * xx_ML + x_pi_mid_k * xx_k + x_pi_mid_L * xx_L

        x_pl_mid_wp = self.fc3_pl_wp(x); x_pl_mid_p = self.fc3_pl_p(x); x_pl_mid_sp = self.fc3_pl_sp(x)
        x_pl_mid = torch.cat([x_pl_mid_wp, x_pl_mid_p, x_pl_mid_sp], dim=1)
        x_pl_mid = F.relu(self.fcc3_pl(x_pl_mid))
        x_pl_weight = F.softmax(self.fc3_pl_weight(x_pl_mid), dim=1)
        xx_wp = x_pl_weight[:, 0].view(x_pl_weight.shape[0], 1).repeat(1, x_pl_mid_wp.shape[1])
        xx_p = x_pl_weight[:, 1].view(x_pl_weight.shape[0], 1).repeat(1, x_pl_mid_p.shape[1])
        xx_sp = x_pl_weight[:, 2].view(x_pl_weight.shape[0], 1).repeat(1, x_pl_mid_sp.shape[1])
        x_pl_mid_fi = x_pl_mid_wp * xx_wp + x_pl_mid_p * xx_p + x_pl_mid_sp * xx_sp


        x_pi = torch.cat([x_pi_mid, x_pl_mid_fi], dim=1)
        x_pi = F.relu(x_pi).view(batch, 2, 128)
        x_pi = self.trans_block_pi(x_pi).view(batch, -1)
        x_pi = self.fc4_pi(x_pi)
        output_pi = x_pi

        # positive level regreesion task
        x_pl = torch.cat([x_pl_mid, x_pi_mid_fi], dim=1)
        x_pl = F.relu(x_pl).view(batch, 2, 128)
        x_pl = self.trans_block_pl(x_pl).view(batch, -1)
        x_pl = self.fc4_pl(x_pl)
        output_pl = torch.sigmoid(x_pl)

        return output_pl, output_pi

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

def ordinal_loss(output, target, device):
    target = target.long()
    target_converted = torch.zeros(output.shape, device=device)
    for i in range(output.shape[0]):
        target_converted[i, :target[i]] = 1
    loss = torch.mean((output - target_converted)**2)
    return loss

def ordinal_predicted(output, device):
    output = torch.round(output)
    compared = torch.cat([torch.ones(output.shape[0], 1, device=device), output[:,:-1]], axis=1)
    predicted = torch.sum(torch.where(output<=compared, output, compared), (1))
    return predicted

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

loss_func_pi = nn.CrossEntropyLoss()

repeat = 1
r_accs = np.zeros((repeat, 6))
unique_labels = {"弱阳性": 0, "阳性": 1, "强阳性": 2}
num_of_types = 3

unique_labels_pi = {"IgAk": 0, "IgAL": 1, "IgGk": 2, "IgGL": 3, "IgMk": 4, "IgML": 5, "k": 6, "L": 7}
num_of_types_pi = 8

for r in range(repeat):
    print("repeating #" + str(r))
    model = Net()
    for param in model.parameters():
        param.requires_grad = True
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=20)

    losses_lst = []
    accs_pl_lst = []
    accs_pi_lst = []
    best_valid_placc = 0; best_valid_piacc = 0
    for epoch in range(0, 400):

        #model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs_pl = AverageMeter()
        accs_pi = AverageMeter()

        tic = time.time()
        model.train()
        with tqdm(total=num_train) as pbar:
            for i, (x, y_pl, y_pi) in enumerate(train_loader):
                optimizer.zero_grad()
                x, y_pl, y_pi = x.to(device), y_pl.to(device), y_pi.to(device)

                # initialize location vector and hidden state
                batch_size = x.shape[0]
                output_pl, output_pi = model(x)
                predicted_pl = ordinal_predicted(output_pl, device)
                _, predicted_pi = torch.max(output_pi, 1)
                loss = 2.2*ordinal_loss(output_pl, y_pl, device) + loss_func_pi(output_pi, y_pi)

                # compute accuracy
                correct_pl = torch.sum((predicted_pl == y_pl).float())
                acc_pl = 100 * (correct_pl / batch_size)
                correct_pi = torch.sum((predicted_pi == y_pi).float())
                acc_pi = 100 * (correct_pi / batch_size)

                # store
                losses.update(loss.item(), x.size()[0])
                accs_pl.update(acc_pl.item(), x.size()[0])
                accs_pi.update(acc_pi.item(), x.size()[0])

                loss.backward()
                optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(("{:.1f}s - loss: {:.3f} - acc_pl: {:.3f} - acc_pi: {:.3f}".format((toc - tic), losses.avg, accs_pl.avg, accs_pi.avg)))
                pbar.update(batch_size)
                torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            losses = AverageMeter()
            accs_pl = AverageMeter()
            accs_pi = AverageMeter()

            for i, (x, y_pl, y_pi) in enumerate(valid_loader):

                optimizer.zero_grad()

                x, y_pl, y_pi = x.to(device), y_pl.to(device), y_pi.to(device)

                # initialize location vector and hidden state
                batch_size = x.shape[0]
                output_pl, output_pi = model(x)
                predicted_pl = ordinal_predicted(output_pl, device)
                _, predicted_pi = torch.max(output_pi, 1)
                loss = 2.2*ordinal_loss(output_pl, y_pl, device) + loss_func_pi(output_pi, y_pi)

                # compute accuracy
                correct_pl = torch.sum((predicted_pl == y_pl).float())
                acc_pl = 100 * (correct_pl / batch_size)
                correct_pi = torch.sum((predicted_pi == y_pi).float())
                acc_pi = 100 * (correct_pi / batch_size)

                # store
                losses.update(loss.item(), x.size()[0])
                accs_pl.update(acc_pl.item(), x.size()[0])
                accs_pi.update(acc_pi.item(), x.size()[0])

            losses_lst.append(losses.avg)
            accs_pl_lst.append(accs_pl.avg)
            accs_pi_lst.append(accs_pl.avg)

            print("validation losses avg. " + str(losses.avg))
            print("validation accs_pl avg. " + str(accs_pl.avg))
            print("validation accs_pi avg. " + str(accs_pi.avg))
            print("epoch: " + str(epoch + 1))

            torch.cuda.empty_cache()

        scheduler.step(-accs_pl.avg)

        # Evaluate on validation set for model saving
        with torch.no_grad():
            correct_pi = 0; correct_pl = 0

            correct_dict = dict(zip(range(num_of_types), [0] * num_of_types))
            total_dict = dict(zip(range(num_of_types), [0] * num_of_types))
            wrong_matrix = np.zeros((num_of_types, num_of_types))

            correct_dict_pi = dict(zip(range(num_of_types_pi), [0] * num_of_types_pi))
            total_dict_pi = dict(zip(range(num_of_types_pi), [0] * num_of_types_pi))
            wrong_matrix_pi = np.zeros((num_of_types_pi, num_of_types_pi))

            start = time.time()
            pred_label_pl = []; true_label_pl = []
            pred_label_pi = []; true_label_pi = []

            model.eval()

            for i, (x, y_pl, y_pi) in enumerate(valid_loader):
                x, y_pl, y_pi = x.to(device), y_pl.to(device), y_pi.to(device)

                batch_size = x.shape[0]
                output_pl, output_pi = model(x)
                predicted_pl = ordinal_predicted(output_pl, device)
                predicted_pl = predicted_pl.long()
                pred_label_pl.append(predicted_pl.cpu().detach().numpy().tolist())
                true_label_pl.append(y_pl.long().cpu().detach().numpy().tolist())

                _, predicted_pi = torch.max(output_pi, 1)
                predicted_pi = predicted_pi.long()
                pred_label_pi.append(predicted_pi.cpu().detach().numpy().tolist())
                true_label_pi.append(y_pi.long().cpu().detach().numpy().tolist())

                correct_pl += torch.sum((predicted_pl == y_pl).float())
                correct_pi += torch.sum((predicted_pi == y_pi).float())

                for j in range(y_pl.data.size()[0]):
                    total_dict[y_pl.data[j].item()] = total_dict[y_pl.data[j].item()] + 1
                    if (predicted_pl[j] == y_pl.data[j]).item():
                        correct_dict[y_pl.data[j].item()] = correct_dict[y_pl.data[j].item()] + 1
                    else:
                        wrong_matrix[y_pl.data[j].item(), predicted_pl[j].item()] += 1
                for j in range(y_pi.data.size()[0]):
                    total_dict_pi[y_pi.data[j].item()] = total_dict_pi[y_pi.data[j].item()] + 1
                    if (predicted_pi[j] == y_pi.data[j]).item():
                        correct_dict_pi[y_pi.data[j].item()] = correct_dict_pi[y_pi.data[j].item()] + 1
                    else:
                        wrong_matrix_pi[y_pi.data[j].item(), predicted_pi[j].item()] += 1

            end = time.time()

            perc_pl = (100.0 * correct_pl) / (num_valid)
            error_pl = 100 - perc_pl
            perc_pi = (100.0 * correct_pi) / (num_valid)
            error_pi = 100 - perc_pi

            print("[*] Valid_pl Acc: {}/{} ({:.6f}% - {:.6f}%)".format(correct_pl, num_valid, perc_pl, error_pl))
            print("[*] Valid_pi Acc: {}/{} ({:.6f}% - {:.6f}%)".format(correct_pi, num_valid, perc_pi, error_pi))

            true_label_pl = [element for sublist in true_label_pl for element in sublist]
            pred_label_pl = [element for sublist in pred_label_pl for element in sublist]
            true_label_pi = [element for sublist in true_label_pi for element in sublist]
            pred_label_pi = [element for sublist in pred_label_pi for element in sublist]

            Preci_pl = precision_score(true_label_pl, pred_label_pl, average='macro')
            Recal_pl = recall_score(true_label_pl, pred_label_pl, average='macro')
            F1score_pl = f1_score(true_label_pl, pred_label_pl, average='macro')
            print('PL Valid Accuracy {:.6f} | PL Valid F1-score {:.6f} | PL Valid Recall {:.6f} | PL Valid Presision {:.6f}'.format(
                    perc_pl, F1score_pl, Recal_pl, Preci_pl))
            Preci_pi = precision_score(true_label_pi, pred_label_pi, average='macro')
            Recal_pi = recall_score(true_label_pi, pred_label_pi, average='macro')
            F1score_pi = f1_score(true_label_pi, pred_label_pi, average='macro')
            print('PI Valid Accuracy {:.6f} | PI Valid F1-score {:.6f} | PI Valid Recall {:.6f} | PI Valid Presision {:.6f}'.format(
                    perc_pi, F1score_pi, Recal_pi, Preci_pi))
            if perc_pi > best_valid_piacc and perc_pl > 50:
                best_valid_placc = perc_pl; best_valid_piacc = perc_pi
                print('*******Now Best Performance!!!!*******')
                print('PL Valid Accuracy {:.6f} | PL Valid F1-score {:.6f} | PL Valid Recall {:.6f} | PL Valid Presision {:.6f}'.format(
                        perc_pl, F1score_pl, Recal_pl, Preci_pl))
                print('PI Valid Accuracy {:.6f} | PI Valid F1-score {:.6f} | PI Valid Recall {:.6f} | PI Valid Presision {:.6f}'.format(
                        perc_pi, F1score_pi, Recal_pi, Preci_pi))
                print('*******Please Come on!!!!*******')
                for label in unique_labels.keys():
                    print("Type: " + label + "\t Total: " + str(
                        total_dict[unique_labels[label]]) + "\t Correct Ratio: " + str(
                        round(correct_dict[unique_labels[label]] / (total_dict[unique_labels[label]]+1e-17), 6)))
                for label in unique_labels_pi.keys():
                    print("Type: " + label + "\t Total: " + str(
                        total_dict_pi[unique_labels_pi[label]]) + "\t Correct Ratio: " + str(
                        round(correct_dict_pi[unique_labels_pi[label]] / (total_dict_pi[unique_labels_pi[label]]+1e-17), 6)))
                torch.save({'model_state_dict': model.state_dict()},
                           '.../Net_SCI_' + str(int(r)) + '.pt')
                sns.set()
                C_pl = confusion_matrix(true_label_pl, pred_label_pl, labels=[0, 1, 2])
                print('Confusion Matrix of Positive Level:', C_pl)
                C_pi = confusion_matrix(true_label_pi, pred_label_pi, labels=[0, 1, 2, 3, 4, 5, 6, 7])
                print('Confusion Matrix of Positive Isotype:', C_pi)

            elif perc_pi == best_valid_piacc and perc_pl > 50:
                if perc_pl > best_valid_placc:
                    best_valid_placc = perc_pl
                    print('*******Now Best Performance!!!!*******')
                    print('PL Valid Accuracy {:.6f} | PL Valid F1-score {:.6f} | PL Valid Recall {:.6f} | PL Valid Presision {:.6f}'.format(
                            perc_pl, F1score_pl, Recal_pl, Preci_pl))
                    print('PI Valid Accuracy {:.6f} | PI Valid F1-score {:.6f} | PI Valid Recall {:.6f} | PI Valid Presision {:.6f}'.format(
                            perc_pi, F1score_pi, Recal_pi, Preci_pi))
                    print('*******Please Come on!!!!*******')
                    for label in unique_labels.keys():
                        print("Type: " + label + "\t Total: " + str(
                            total_dict[unique_labels[label]]) + "\t Correct Ratio: " + str(
                            round(correct_dict[unique_labels[label]] / (total_dict[unique_labels[label]]+1e-17), 6)))
                    for label in unique_labels_pi.keys():
                        print("Type: " + label + "\t Total: " + str(
                            total_dict_pi[unique_labels_pi[label]]) + "\t Correct Ratio: " + str(
                            round(correct_dict_pi[unique_labels_pi[label]] / (total_dict_pi[unique_labels_pi[label]]+1e-17), 6)))
                    torch.save({'model_state_dict': model.state_dict()},
                               '.../Net_SCI_' + str(int(r)) + '.pt')
                    sns.set()
                    C_pl = confusion_matrix(true_label_pl, pred_label_pl, labels=[0, 1, 2])
                    print('Confusion Matrix of Positive Level:', C_pl)
                    C_pi = confusion_matrix(true_label_pi, pred_label_pi, labels=[0, 1, 2, 3, 4, 5, 6, 7])
                    print('Confusion Matrix of Positive Isotype:', C_pi)
