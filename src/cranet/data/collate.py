import cranet
from collections import Sequence, Mapping


def default_collate_fn(batch):
    elm = batch[0]
    elm_type = type(elm)
    if isinstance(elm, cranet.Tensor):
        return cranet.stack(batch, 0)
    elif isinstance(elm, float):
        return cranet.tensor(batch, dtype=cranet.float64)
    elif isinstance(elm, int):
        return cranet.tensor(batch)
    elif isinstance(elm, str):
        return
    elif isinstance(elm, Mapping):
        return {key: default_collate_fn([d[key] for d in batch]) for key in elm}
    elif isinstance(elm, tuple) and hasattr(elm, '_fields'):  # namedtuple
        return elm_type(*(default_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elm, Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate_fn(samples) for samples in transposed]
    raise TypeError


def default_convert_fn():
    pass
