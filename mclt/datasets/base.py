import abc
from functools import cached_property
from typing import Any, Callable, Dict, Optional

from datasets import Dataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast

AugmentationType = Callable[[dict[str, Any]], dict[str, Any]]


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

    @cached_property
    def label_mapping(self) -> Dict[int, Any]:
        dataset = self._train_dataset
        label_cols = [col for col in dataset.column_names if col.startswith('label')]
        if len(label_cols) == 1:
            labels = set(dataset[label_cols[0]])
        else:
            labels = {False: 0, True: 1}
        return {name: i for i, name in enumerate(labels)}

    @property
    def num_labels(self) -> int:
        return len(self.label_mapping)

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
                label_int = {'label': [self.label_mapping[label] for label in labels]}
            else:
                label_int = {
                    col: [self.label_mapping[bool(label)] for label in items[col]]
                    for col in label_cols
                }

            return {
                **label_int,
                **tokenized,
            }

        text_cols = {col for col in dataset.column_names if col.startswith('text')}
        label_cols = {col for col in dataset.column_names if col.startswith('label')}

        dataset.set_transform(process_batch, output_all_columns=True)
        return dataset

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = self._train_dataset
        # if self._train_sample_size:
        #     dataset = dataset.shuffle(seed=self._seed)[:self._train_sample_size]
        return self._create_dataloader(self._preprocess(dataset), shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._preprocess(self._val_dataset), shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._preprocess(self._test_dataset), shuffle=False)

    def _create_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            collate_fn=DataCollatorWithPadding(
                tokenizer=self._tokenizer,
                padding='longest',
            ),
            shuffle=shuffle,
        )
