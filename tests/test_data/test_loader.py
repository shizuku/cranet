import cranet
from cranet.data import Dataset, DataLoader

import numpy as np


class TestDataset(Dataset):
    def __len__(self):
        return 6000

    def __getitem__(self, item):
        img = np.random.rand(1, 28, 28)
        lab = np.random.randint(0, 10)
        return cranet.as_tensor(img), cranet.as_tensor(lab)


train_ds = TestDataset()

train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)

for img, lab in train_ld:
    print(img.shape, lab.shape)
