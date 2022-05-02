import json
import os
import warnings
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from transformers import AutoModel, AutoTokenizer

from mclt import DEFAULT_MLFLOW_TRACKING_URI, PROJECT_PATH, EMBEDDINGS_DIR
from mclt.data import DATASETS, TASK_LANG_MATRIX
from mclt.data.base import MultiTaskDataModule, TaskDefinition
from mclt.modeling.multi_classifier import MultiTaskTransformer
from mclt.training import LOSS_FUNCS
from mclt.training.trainer import MultitaskTransformerTrainer
from mclt.utils.seed import set_seeds


def run_experiment(
    config: dict[str, Any],
    create_datamodule: Callable[[Dict], MultiTaskDataModule],
    create_model_trainer: Callable[[Dict, dict[str, TaskDefinition]], MultitaskTransformerTrainer],
    experiment_name: str,
    experiment_tag: str,
):
    seed = config['model_random_state']
    task_names = '_'.join(config['tasks']) if config['tasks'] else 'multitask'
    lang_names = '_'.join(config['languages']) if config['languages'] else 'multilang'
    matrics_log_path = PROJECT_PATH.joinpath(
        'data',
        'metrics',
        experiment_name,
        f'{task_names}|{lang_names}',
        f'{seed}.json',
    )
    if matrics_log_path.exists():
        warnings.warn(f"Results for {experiment_name}, f'{task_names}|{lang_names}', with seed {seed} already exist")
        return

    matrics_log_path.parent.mkdir(exist_ok=True, parents=True)

    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', DEFAULT_MLFLOW_TRACKING_URI)
    logger = MLFlowLogger(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        tags={'description': experiment_tag},
    )
    logger.log_hyperparams(config)

    checkpoint_path = (
        Path(mlflow_tracking_uri)
        .joinpath(
            'checkpoints',
            logger.experiment_id,
            logger.run_id,
        )
        .with_suffix('.ckpt')
    )
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)

    trainer = Trainer(
        max_steps=config['max_steps'],
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_path.parent,
                filename=f'{checkpoint_path.stem}',
                monitor='val_set/f1_score',
                mode='max',
            ),
            LearningRateMonitor('step'),
        ],
        gpus=config['gpus'],
        precision=16 if config['gpus'] else 32,
        accumulate_grad_batches=config['accumulate_grad_batches'],
        val_check_interval=config['val_check_interval'],
    )

    datamodule = create_datamodule(config)
    model_trainer = create_model_trainer(config, datamodule.tasks)
    trainer.fit(
        model=model_trainer,
        datamodule=datamodule,
    )

    metrics, *_ = trainer.test(
        dataloaders=datamodule.val_dataloader(),
    )
    logger.log_metrics({f'test/{k}': v for k, v in metrics.items()})

    with matrics_log_path.open('w') as file:
        json.dump(metrics, file, indent=2)

    checkpoint_path.unlink(missing_ok=True)


def create_datamodule(config) -> MultiTaskDataModule:
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    kwargs = {
        'tokenizer': tokenizer,
        'batch_size': config['batch_size'],
        'num_workers': config['num_workers'],
        'seed': config['data_random_state'],
        'max_length': config['max_length'],
        'train_sample_size': config['train_sample_size'],
    }

    datasets = TASK_LANG_MATRIX.loc[
        config['languages'] or slice(None),
        config['tasks'] or slice(None),
    ]
    datasets = list(chain.from_iterable(datasets.values))
    datamodule = MultiTaskDataModule(
        [DATASETS[dataset](**kwargs) for dataset in datasets],
        tokenizer=tokenizer,
        num_workers=config['num_workers'],
        batch_size=config['batch_size'],
    )
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def create_multilingual_model_trainer(config, tasks: dict[str, TaskDefinition]):
    set_seeds(config['model_random_state'])

    transformer = AutoModel.from_pretrained(
        config['model'],
    )

    loss_func = LOSS_FUNCS[config['loss_func']]
    model = MultiTaskTransformer(
        transformer=transformer,
        tasks=tasks,
        loss_func=loss_func(tasks),
    )

    if config['gradient_checkpointing']:
        transformer.gradient_checkpointing_enable()

    freeze_backbone = config['freeze_backbone']
    if isinstance(freeze_backbone, bool) and freeze_backbone:
        transformer.base_model.apply(lambda param: param.requires_grad_(False))
        transformer.forward = EmbeddingSaver(transformer.forward)
    elif isinstance(freeze_backbone, str) and freeze_backbone == 'embeddings':
        transformer.base_model.embeddings.apply(lambda param: param.requires_grad_(False))

    return MultitaskTransformerTrainer(
        model=model,
        tasks=tasks,
        learning_rate=config['learning_rate'],
        warmup_steps_ratio=config['warmup_steps_ratio'],
    )


class EmbeddingSaver:
    def __init__(self, forward, name='default'):
        self._forward = forward
        self._store = EMBEDDINGS_DIR / name
        self._store.mkdir(exist_ok=True, parents=True)
        self._index_path = self._store.joinpath('index.pkl')
        self._index = torch.load(self._index_path) if self._index_path.exists() else {}

    def __call__(self, input_ids, attention_mask):
        keys = self._get_keys(input_ids, attention_mask)
        lengths = [mask.sum() for mask in attention_mask]
        not_cached_ids = []
        outputs = [None for _ in input_ids]
        for i, key in enumerate(keys):
            index = self._index.get(key)
            if index is not None:
                emb_path = self._store.joinpath(index)
                outputs[i] = torch.tensor(torch.load(emb_path), device=input_ids.device)
            else:
                not_cached_ids.append(i)

        if not_cached_ids:
            input_ids = input_ids[not_cached_ids]
            attention_mask = attention_mask[not_cached_ids]
            output = self._forward(input_ids, attention_mask).last_hidden_state
            for i, o in zip(not_cached_ids, output):
                key = keys[i]
                outputs[i] = o
                index = str(len(self._index))
                self._index[key] = index
                emb_path = self._store.joinpath(index)
                o = o[:lengths[i]]
                self._safe_save(
                    o.detach().cpu().numpy(),
                    emb_path,
                )

            self._safe_save(self._index, self._index_path)

        return self._pad_to_longest(outputs)

    @staticmethod
    def _get_keys(input_ids, attention_mask):
        keys = []
        for ids, mask in zip(input_ids, attention_mask):
            length = mask.sum()
            key = ','.join(map(str, ids[:length].tolist()))
            keys.append(key)
        return keys

    @staticmethod
    def _pad_to_longest(outputs):
        max_len = max(map(len, outputs))
        o = outputs[0]
        tensor = torch.ones(len(outputs), max_len, o.shape[1], device=o.device)

        for o, t in zip(outputs, tensor):
            t[:len(o)] = o

        return tensor

    @staticmethod
    def _safe_save(obj, path: Path):
        temp_path = path.with_suffix('.tmp')
        torch.save(obj, temp_path)
        temp_path.rename(path)
