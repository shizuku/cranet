import random
from typing import TypeVar, Generic, Sized, Iterator, Union, Iterable, List

T_co = TypeVar('T_co', covariant=True)


class Sampler(Generic[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError


class SequentialSampler(Sampler[int]):
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        super().__init__()
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    data_source: Sized

    def __init__(self, data_source: Sized):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        l = list(range(len(self.data_source)))
        random.shuffle(l)
        return iter(l)

    def __len__(self) -> int:
        return len(self.data_source)


class BatchSampler(Sampler):
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
