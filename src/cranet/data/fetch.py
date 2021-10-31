class DatasetFetcher:
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.dataset_iter = iter(dataset)
        self.ended = False

    def fetch(self, slice):
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
            for _ in slice:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    self.ended = True
                    break
            if len(data) == 0 or (self.drop_last and len(data) < len(slice)):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)
