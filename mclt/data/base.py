import abc
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Dict, Optional, Type

import numpy as np
import torch
from datasets import Dataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast

from mclt.data.dataset import MultiTaskDataset
from mclt.utils.collator import CustomDataCollatorWithPadding

AugmentationType = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class TaskDefinition:
    num_labels: int
    multilabel: bool


class BaseDataModule(LightningDataModule, abc.ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        augmentations: Optional[AugmentationType] = None,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = 256,
        train_sample_size: Optional[int] = 10000,
    ):
        super().__init__()
        self.augmentations = augmentations
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._seed = seed
        self._max_length = max_length
        self._tokenizer = tokenizer
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._train_sample_size = train_sample_size

    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    def train_dataset(self):
        return self._preprocess(self._train_dataset)

    @property
    def val_dataset(self):
        return self._preprocess(self._val_dataset)

    @property
    def test_dataset(self):
        return self._preprocess(self._test_dataset)

    @cached_property
    def label_mapping(self) -> Dict[int, Any]:
        dataset = self._train_dataset
        label_cols = [col for col in dataset.column_names if col.startswith('label')]
        if len(label_cols) == 1:
            labels = set(dataset[label_cols[0]])
        else:
            labels = label_cols
        return {name: i for i, name in enumerate(labels)}

    @property
    def num_labels(self) -> int:
        return len(self.label_mapping)

    @property
    def multilabel(self) -> bool:
        return False

    def _preprocess(self, dataset):
        def process_batch(items):
            tokenized = self._tokenizer(
                *(items[col] for col in text_cols),
                truncation=True,
                return_tensors='pt',
                max_length=self._max_length,
            )
            if len(label_cols) == 1:
                labels = items['label']
                labels_int = [self.label_mapping[label] for label in labels]
            else:
                labels_int = []
                for i in range(tokenized.n_sequences):
                    row = [items[col][i] for col in label_cols]
                    labels_int.append(torch.tensor(row, dtype=torch.float32))

            return {
                'labels': labels_int,
                'task': [self.name],
                **tokenized,
            }

        if self._train_sample_size:
            dataset = self._subsample_dataset(dataset)

        text_cols = {col for col in dataset.column_names if col.startswith('text')}
        label_cols = [col for col in dataset.column_names if col.startswith('label')]

        dataset.set_transform(process_batch, output_all_columns=True)
        return dataset

    def _subsample_dataset(self, dataset: Dataset):
        np.random.seed(self._seed)
        indices = np.random.permutation(len(dataset))[: self._train_sample_size]
        return dataset.select(indices)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._create_dataloader(
            self.train_dataset,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(
            self._val_dataset,
            shuffle=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(
            self._test_dataset,
            shuffle=False,
        )

    def _create_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            collate_fn=CustomDataCollatorWithPadding(
                tokenizer=self._tokenizer,
                padding='longest',
            ),
            shuffle=shuffle,
        )


class MultiTaskDataModule(LightningDataModule):
    def __init__(
        self,
        datamodules: list[BaseDataModule],
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int = 16,
        num_workers: int = 8,
    ):
        super().__init__()

        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._datamodules = sorted(datamodules, key=lambda d: d.name)
        self._train_dataset = None
        self._val_datasets = None
        self._test_datasets = None

    @property
    def name(self) -> str:
        return ' + '.join(d.name for d in self._datamodules)

    @property
    def tasks(self) -> dict[str, TaskDefinition]:
        return {d.name: TaskDefinition(d.num_labels, d.multilabel) for d in self._datamodules}

    def prepare_data(self) -> None:
        for datamodule in self._datamodules:
            datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        train_datasets, val_datasets, test_datasets, names = [], [], [], []
        for datamodule in self._datamodules:
            datamodule.setup(stage=stage)
            train_datasets.append(datamodule.train_dataset)
            val_datasets.append(datamodule.val_dataset)
            test_datasets.append(datamodule.test_dataset)
            names.append(datamodule.name)

        self._train_dataset = MultiTaskDataset(
            datasets={name: dataset for name, dataset in zip(names, train_datasets)}
        )
        self._val_datasets = val_datasets
        self._test_datasets = test_datasets

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._create_dataloader(
            self._train_dataset,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [
            self._create_dataloader(
                dataset,
            )
            for dataset in self._val_datasets
        ]

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return [
            self._create_dataloader(
                dataset,
            )
            for dataset in self._test_datasets
        ]

    def _create_dataloader(
        self,
        dataset: Dataset,
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            collate_fn=CustomDataCollatorWithPadding(
                tokenizer=self._tokenizer,
                padding='longest',
            ),
        )
