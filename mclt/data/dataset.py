from random import shuffle
from typing import Iterable, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset, IterableDataset


class MultiTaskDataset(IterableDataset):
    def __init__(
        self,
        datasets: dict[str, Dataset],
        weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        if weights:
            assert len(weights) == len(
                datasets
            ), f'len(weights) = {len(weights)}, len(datasets) = {len(datasets)}'
        else:
            weights = [1.0 / len(datasets)] * len(datasets)

        self._datasets = [shuffle_cycle(dataset) for dataset in datasets.values()]
        self._weights = weights

    def __iter__(self):
        while True:
            dataset = np.random.choice(
                self._datasets,
                p=self._weights,
            )
            yield next(dataset)


def shuffle_cycle(sequence: Dataset) -> Iterable:
    ids = list(range(len(sequence)))  # type: ignore
    while True:
        shuffle(ids)
        for idx in ids:
            yield sequence[idx]
