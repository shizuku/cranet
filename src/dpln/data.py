from typing import List, TypeVar

import numpy as np

T = TypeVar('T')


class Dataset:
    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def enumerate(self, start=0):
        idx = list(range(start, start + len(self)))
        return ZipDataset(self, ListDataset(idx))


class ListDataset(Dataset):
    def __init__(self, x: List[T]):
        self.x = x
        self.idx = 0

    def batch(self, batch_size: int):
        r = []
        for i in range(0, len(self.x), batch_size):
            tx = self.x[i:i + batch_size]
            r.append(np.array(tx))
        return ListDataset(r)

    def __getitem__(self, index: int):
        return self.x[index]

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self):
            raise StopIteration
        else:
            r = self[self.idx]
            self.idx += 1
            return r


class ZipDataset(Dataset):
    def __init__(self, x: Dataset, y: Dataset):
        self.x = x
        self.y = y
        assert len(self.x) == len(self.y)
        self.idx = 0

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self):
            raise StopIteration
        else:
            r = self[self.idx]
            self.idx += 1
            return r
