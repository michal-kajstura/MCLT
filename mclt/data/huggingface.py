import abc
from itertools import chain
from typing import Literal, Optional, Union

import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from transformers import PreTrainedTokenizerFast

from mclt import DATASETS_PATH
from mclt.data.base import BaseDataModule
from mclt.utils.seed import set_seeds


class GivenSplitsMixin:
    def _load_splits(self, dataset: Dataset) -> tuple[Dataset, Dataset, Dataset]:
        train = dataset['train']
        val = dataset['validation']
        test = dataset['test']
        return train, val, test


class RandomSplitMixin:
    _train_size: float = 0.8
    _seed: int

    def _load_splits(self, dataset: Dataset) -> tuple[Dataset, Dataset, Dataset]:
        if isinstance(dataset, DatasetDict):
            dataset = dataset['train']

        train_test_val = dataset.train_test_split(train_size=self._train_size, seed=self._seed)
        test_valid = train_test_val['test'].train_test_split(train_size=0.5)

        return train_test_val['train'], test_valid['train'], test_valid['test']


class MonolingualLoadMixin:
    name: str

    def _load_dataset(self):
        return datasets.load_dataset(self.name)


class MultilingualLoadMixin:
    name: str
    _language: str

    def _load_dataset(self):
        return datasets.load_dataset(self.name, self._language)


class BaseHuggingfaceDataModule(GivenSplitsMixin, MonolingualLoadMixin, BaseDataModule, abc.ABC):
    def setup(self, stage: Optional[str] = None) -> None:
        set_seeds(self._seed)
        dataset = self._load_dataset()
        train, val, test = self._load_splits(dataset)
        self._train_dataset = self._process_dataset(train)
        self._val_dataset = self._process_dataset(val)
        self._test_dataset = self._process_dataset(test)

    def _process_dataset(self, dataset: Dataset):
        text_columns = self._column_mapping['text']
        mappings = {}
        if isinstance(text_columns, str):
            mappings[text_columns] = 'text'
        else:
            for i, item in enumerate(text_columns):
                mappings[item] = f'text_{i}'

        label = self._column_mapping['label']
        if isinstance(label, str):
            mappings[label] = 'label'
        else:
            for i, item in enumerate(label):
                mappings[item] = f'label_{i}'

        dataset = dataset.rename_columns(mappings)
        return dataset.remove_columns(set(dataset.column_names).difference(mappings.values()))

    @property
    @abc.abstractmethod
    def _column_mapping(self) -> dict[str, str]:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass


class BasePolEmo2DataModule(BaseHuggingfaceDataModule, abc.ABC):
    @property
    def _column_mapping(self) -> dict[str, Union[str, list[str]]]:
        return {
            'text': 'sentence',
            'label': 'target',
        }


class PolEmo2InDataModule(BasePolEmo2DataModule):
    @property
    def name(self) -> str:
        return 'allegro/klej-polemo2-in'


class PolEmo2OutDataModule(BasePolEmo2DataModule):
    @property
    def name(self) -> str:
        return 'allegro/klej-polemo2-out'


class AllegroReviewsDataModule(BaseHuggingfaceDataModule):
    @property
    def name(self) -> str:
        return 'allegro/klej-allegro-reviews'

    @property
    def _column_mapping(self) -> dict[str, Union[str, list[str]]]:
        return {
            'text': 'text',
            'label': 'rating',
        }


class CyberbullyingDetectionDataModule(RandomSplitMixin, BaseHuggingfaceDataModule):
    @property
    def name(self) -> str:
        return 'allegro/klej-cbd'

    @property
    def _column_mapping(self) -> dict[str, Union[str, list[str]]]:
        return {
            'text': 'sentence',
            'label': 'target',
        }


class CDSCEntailmentDataModule(BaseHuggingfaceDataModule):
    @property
    def name(self) -> str:
        return 'allegro/klej-cdsc-e'

    @property
    def _column_mapping(self) -> dict[str, Union[str, list[str]]]:
        return {
            'text': ['sentence_A', 'sentence_B'],
            'label': 'entailment_judgment',
        }


class TweetsHateSpeechDetectionDataModule(RandomSplitMixin, BaseHuggingfaceDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        train_size: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
        train_sample_size: Optional[int] = 10000,
    ):
        super().__init__(
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
            train_sample_size=train_sample_size,
        )
        self._train_size = train_size

    @property
    def name(self) -> str:
        return 'tweets_hate_speech_detection'

    @property
    def _column_mapping(self) -> dict[str, str]:
        return {
            'text': 'tweet',
            'label': 'label',
        }

    def _load_dataset(self) -> None:
        return super()._load_dataset()['train']


class HateSpeech18DataModule(RandomSplitMixin, BaseHuggingfaceDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        train_size: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
        train_sample_size: Optional[int] = 10000,
    ):
        super().__init__(
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
            train_sample_size=train_sample_size,
        )
        self._train_size = train_size

    @property
    def name(self) -> str:
        return 'hate_speech18'

    @property
    def _column_mapping(self) -> dict[str, str]:
        return {
            'text': 'text',
            'label': 'label',
        }

    def _load_dataset(self) -> None:
        return super()._load_dataset()['train']


class HateSpeechOffensiveDataModule(RandomSplitMixin, BaseHuggingfaceDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        train_size: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
        train_sample_size: Optional[int] = 10000,
    ):
        super().__init__(
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
            train_sample_size=train_sample_size,
        )
        self._train_size = train_size

    @property
    def name(self) -> str:
        return 'hate_speech_offensive'

    @property
    def _column_mapping(self) -> dict[str, str]:
        return {
            'text': 'tweet',
            'label': 'class',
        }


class HateSpeechPLDataModule(RandomSplitMixin, BaseHuggingfaceDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        train_size: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
        train_sample_size: Optional[int] = 10000,
    ):
        super().__init__(
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
            train_sample_size=train_sample_size,
        )
        self._train_size = train_size

    @property
    def name(self) -> str:
        return 'hate_speech_pl'

    @property
    def _column_mapping(self) -> dict[str, str]:
        return {
            'text': 'text',
            'label': 'rating',
        }


class AmazonReviewsDataModule(MultilingualLoadMixin, BaseHuggingfaceDataModule):
    def __init__(
        self,
        language: Literal['en', 'de', 'es', 'fr', 'ja', 'zn'],
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
        train_sample_size: Optional[int] = 10000,
    ):
        super().__init__(
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
            train_sample_size=train_sample_size,
        )
        self._language = language

    @property
    def name(self) -> str:
        return 'amazon_reviews_multi'

    @property
    def _column_mapping(self) -> dict[str, str]:
        return {
            'text': 'review_body',
            'label': 'stars',
        }


class XNLIDataModule(RandomSplitMixin, BaseHuggingfaceDataModule):
    def __init__(
        self,
        language: Literal['en', 'de', 'es', 'fr', 'ja', 'zn'],
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
        train_sample_size: Optional[int] = 10000,
    ):
        super().__init__(
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
            train_sample_size=train_sample_size,
        )
        self._language = language

    @property
    def name(self) -> str:
        return 'xtreme'

    def _load_dataset(self):
        dataset = datasets.load_dataset(self.name, 'XNLI').filter(
            lambda item: item['language'] == self._language
        )
        return concatenate_datasets([dataset['test'], dataset['validation']])

    @property
    def _column_mapping(self) -> dict[str, str]:
        return {
            'text': ['sentence1', 'sentence2'],
            'label': 'gold_label',
        }


class PANXDataModule(MultilingualLoadMixin, BaseHuggingfaceDataModule):
    def __init__(
        self,
        language: Literal['en', 'de', 'es', 'fr', 'bg', 'nd'],
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
        train_sample_size: Optional[int] = 10000,
    ):
        super().__init__(
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
            train_sample_size=train_sample_size,
        )
        self._language = f'PAN-X.{language}'

    @property
    def name(self) -> str:
        return 'xtreme'

    @property
    def _column_mapping(self) -> dict[str, str]:
        return {
            'premise': 'text',
            'hypothesis': 'hypothesis',
            'label': 'label',
        }


class SemEval2018Task1DataModule(MultilingualLoadMixin, BaseHuggingfaceDataModule):
    def __init__(
        self,
        language: Literal['english', 'german', 'spanish'],
        tokenizer: PreTrainedTokenizerFast,
        train_size: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
        train_sample_size: Optional[int] = 10000,
    ):
        super().__init__(
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
            train_sample_size=train_sample_size,
        )
        self._train_size = train_size
        self._language = f'subtask5.{language}'

    @property
    def name(self) -> str:
        return 'sem_eval_2018_task_1'

    @property
    def _column_mapping(self) -> dict[str, str]:
        return {
            'text': 'Tweet',
            'label': [
                'anger',
                'anticipation',
                'disgust',
                'fear',
                'joy',
                'love',
                'optimism',
                'pessimism',
                'sadness',
                'surprise',
                'trust',
            ],
        }

    @property
    def multilabel(self) -> bool:
        return True


class GoEmotionsDataModule(RandomSplitMixin, BaseHuggingfaceDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        train_size: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
        train_sample_size: Optional[int] = 10000,
    ):
        super().__init__(
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
            train_sample_size=train_sample_size,
        )
        self._train_size = train_size

    def _load_dataset(self):
        return datasets.load_dataset(self.name, 'raw')

    @property
    def name(self) -> str:
        return 'go_emotions'

    @property
    def _column_mapping(self) -> dict[str, str]:
        return {
            'text': 'text',
            'label': [
                'admiration',
                'amusement',
                'anger',
                'annoyance',
                'approval',
                'caring',
                'confusion',
                'curiosity',
                'desire',
                'disappointment',
                'disapproval',
                'disgust',
                'embarrassment',
                'excitement',
                'fear',
                'gratitude',
                'grief',
                'joy',
                'love',
                'nervousness',
                'optimism',
                'pride',
                'realization',
                'relief',
                'remorse',
                'sadness',
                'surprise',
                'neutral',
            ],
        }

    @property
    def multilabel(self) -> bool:
        return True


class MTSCDataModule(RandomSplitMixin, BaseHuggingfaceDataModule):
    _dataset_path = DATASETS_PATH / 'multilingual_twitter_sentiment_classification/tweets'

    def __init__(
        self,
        language: Literal['en', 'de', 'hu', 'pl', 'pt', 'ru', 'sk', 'so', 'es'],
        tokenizer: PreTrainedTokenizerFast,
        train_size: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
        train_sample_size: Optional[int] = 10000,
    ):
        super().__init__(
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
            train_sample_size=train_sample_size,
        )
        self._language = language
        self._train_size = train_size

    def _load_dataset(self):
        return datasets.load_dataset(
            'csv',
            data_files=str(self._dataset_path / f'{self._language}.csv'),
        )['train']

    @property
    def name(self) -> str:
        return f'mclt_{self._language}'

    @property
    def _column_mapping(self) -> dict[str, str]:
        return {
            'text': 'Tweet text',
            'label': 'SentLabel',
        }


class CEDRDataModule(RandomSplitMixin, BaseHuggingfaceDataModule):
    _dataset_path = DATASETS_PATH / 'multilingual_twitter_sentiment_classification/tweets'

    def _load_dataset(self):
        dataset = datasets.load_dataset('cedr')
        dataset = concatenate_datasets([dataset['train'], dataset['test']])
        num_labels = 5

        def onehot(row):
            labels = row.pop('labels')
            for i in range(num_labels):
                row[str(i)] = 1 if i in labels else 0
            return row
        dataset = dataset.map(onehot)
        return dataset

    @property
    def name(self) -> str:
        return 'cedr'

    @property
    def _column_mapping(self) -> dict[str, str]:
        return {
            'text': 'text',
            'label': [
                '0',
                '1',
                '2',
                '3',
                '4',
            ]
        }

    @property
    def multilabel(self) -> bool:
        return True


class RuReviewsDataModule(RandomSplitMixin, BaseHuggingfaceDataModule):
    _dataset_path = DATASETS_PATH / 'rureviews/rureviews.csv'

    def _load_dataset(self):
        return datasets.load_dataset(
            'csv',
            data_files=str(self._dataset_path),
            delimiter='\t',
        )

    @property
    def name(self) -> str:
        return 'rureviews'

    @property
    def _column_mapping(self) -> dict[str, str]:
        return {
            'text': 'review',
            'label': 'sentiment',
        }


class RuUkAbusiveLanguageDataModule(RandomSplitMixin, BaseHuggingfaceDataModule):
    _dataset_path = DATASETS_PATH / 'ru_uk_AbusiveLanguageDataset/labeled.csv'

    def _load_dataset(self):
        return datasets.load_dataset(
            'csv',
            data_files=str(self._dataset_path),
        )

    @property
    def name(self) -> str:
        return 'ru_uk_abusive_language'

    @property
    def _column_mapping(self) -> dict[str, str]:
        return {
            'text': 'text',
            'label': 'abusive',
        }


class HatEvalDataModule(BaseHuggingfaceDataModule):
    _dataset_path = DATASETS_PATH / 'hateval2019'

    def __init__(
        self,
        language: Literal['en', 'es'],
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
        train_sample_size: Optional[int] = 10000,
    ):
        super().__init__(
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
            train_sample_size=train_sample_size,
        )
        self._language = language

    def _load_dataset(self):
        data = {
            'validation' if split == 'dev' else split: datasets.load_dataset(
                'csv',
                data_files=str(self._dataset_path / f'hateval2019_{self._language}_{split}.csv'),
            )['train'] for split in ('train', 'dev', 'test')
        }
        return DatasetDict(data)

    @property
    def name(self) -> str:
        return f'hateval2019_{self._language}'

    @property
    def _column_mapping(self) -> dict[str, str]:
        return {
            'text': 'text',
            'label': 'HS',
        }


class MallCZDataModule(RandomSplitMixin, BaseHuggingfaceDataModule):
    _dataset_path = DATASETS_PATH / 'mallcz'

    def _load_dataset(self):
        label_mapping = {
            'negative': 0,
            'neutral': 1,
            'positive': 2,
        }
        data = []
        for label, label_id in label_mapping.items():
            dataset = datasets.load_dataset(
                'csv',
                data_files=str(self._dataset_path / f'{label}.txt'),
            )['train']
            dataset['label'] = label_id
            data.append(dataset)

        dataset = concatenate_datasets(data)
        return dataset

    @property
    def name(self) -> str:
        return f'mallcz'

    @property
    def _column_mapping(self) -> dict[str, str]:
        return {
            'text': 'text',
            'label': 'label',
        }
