# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import cranet
from cranet import nn, optim
from cranet.nn import functional as F
from cranet.data import Dataset, DataLoader

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

cranet.__version__

# %%
train_mat = loadmat('train_32x32.mat')


# %%
class SvhnDataset(Dataset):
    def __init__(self, mat, transform=None, transform_target=None) -> None:
        super().__init__()
        self.mat = mat
        self.transform = transform
        self.transform_target = transform_target

    def __len__(self):
        return self.mat['X'].shape[3]

    def __getitem__(self, idx):
        img = self.mat['X'][:, :, :, idx]
        lab = self.mat['y'][idx, :]
        if self.transform:
            img = self.transform(img)
        if self.transform_target:
            lab = self.transform_target(lab)
        return img, lab


# %%
def transform(img: np.ndarray):
    img = img.transpose((2, 0, 1)).astype(np.float32)
    return cranet.Tensor(img)


def transform_target(lab: np.ndarray):
    lab = lab.squeeze().astype(np.int64)-1
    return cranet.Tensor([lab])


train_ds = SvhnDataset(train_mat, transform=transform,
                       transform_target=transform_target)


# %%
def batch_fn(p):
    rx = cranet.concat([i[0].reshape(1, 3, 32, 32) for i in p], dim=0)
    ry = cranet.concat([i[1].reshape(1) for i in p], dim=0)
    return rx, ry


train_ld = DataLoader(train_ds, batch_size=64,
                      batch_fn=batch_fn, shuffle=True)

# %%
sample_img, sample_lab = train_ld[0]

# %%
sample_img.shape

# %%
sample_lab.shape


# %%
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.dropout0 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.linear0 = nn.Linear(64 * 8 * 8, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout0(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.flatten(x, start_dim=1)
        x = self.linear0(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear1(x)
        out = F.log_softmax(x, dim=1)
        return out


# %%
model = Model()

# %%
optm = optim.SGD(model.parameters(), 0.1)

# %%
train_loss = []


# %%
def train(epoch: int):
    for i, (inp, lab) in enumerate(train_ld):
        pre = model(inp)
        loss = F.nll_loss(pre, lab)
        optm.zero_grad()
        loss.backward()
        optm.step()
        loss_v = loss.numpy()
        train_loss.append(loss_v)
        print(f"Epoch:{epoch + 1}\t:Step:{i + 1}\tLoss:{loss_v}")


# %%
epochs = 10

# %%
for i in range(epochs):
    train(i)

# %%
