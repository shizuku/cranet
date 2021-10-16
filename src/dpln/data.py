import random

from typing import List, Callable


class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

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


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, batch_fn: Callable = None, shuffle: bool = False):
        if batch_size < 1:
            raise ValueError("batch_size must >= 1")
        if batch_size > 1 and batch_fn is None:
            raise ValueError("batch_fn can't be None when batch_size>1")
        self.dataset: Dataset = dataset
        self.batch_size: int = batch_size
        self.batch_fn = batch_fn
        self.index_list: List[int] = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.index_list)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):
        if self.batch_size == 1:
            return self.dataset[self.index_list[idx]]
        else:
            batch = []
            for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
                batch.append(self.dataset[self.index_list[i]])
            return self.batch_fn(batch)

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
