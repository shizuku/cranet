from __future__ import annotations

from .dataset import Dataset
from .sampler import Sampler, RandomSampler, SequentialSampler, BatchSampler
from .collate import default_collate_fn, default_convert_fn
from .fetch import DatasetFetcher

from typing import TypeVar, Generic, Optional, Union, Iterable, Callable, List, Any, Sequence

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

_collate_fn_t = Callable[[List[T]], Any]


class DataLoader(Generic[T_co]):
    dataset: Dataset[T_co]
    batch_size: Optional[int]
    sampler: Union[Sampler, Iterable]

    def __init__(self, dataset: Dataset[T_co],
                 batch_size: Optional[int] = 1,
                 shuffle: bool = False,
                 sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 collate_fn: Optional[_collate_fn_t] = None,
                 drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        if batch_sampler is not None:
            batch_size = None
            drop_last = False

        if sampler is None:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler

        if collate_fn is None:
            if self.auto_collation:
                collate_fn = default_collate_fn
            else:
                collate_fn = default_convert_fn
        self.collate_fn = collate_fn

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size

    @property
    def auto_collation(self):
        return self.batch_sampler is not None

    @property
    def index_sampler(self):
        if self.auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __iter__(self):
        return DataLoaderIter(self)


class DataLoaderIter:
    def __init__(self, loader: DataLoader) -> None:
        self._dataset = loader.dataset
        self._drop_last = loader.drop_last
        self._index_sampler = loader.index_sampler
        self._auto_collation = loader.auto_collation
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._dataset_fetcher = DatasetFetcher(self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

    def _reset(self):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._dataset_iter = iter(self._dataset)

    def __iter__(self):
        return self

    def _next_index(self):
        return next(self._sampler_iter)

    def _next_data(self):
        index = self._next_index()
        data = self._dataset_fetcher.fetch(index)
        return data

    def __next__(self):
        if self._sampler_iter is None:
            self._reset()
        data = self._next_data()
        self._num_yielded += 1
        return data

    def __len__(self) -> int:
        return len(self._index_sampler)
