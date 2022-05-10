import json
import os
import warnings
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModel, AutoTokenizer

from mclt import (
    DEFAULT_MLFLOW_TRACKING_URI,
    PROJECT_PATH,
)
from mclt.data import DATASETS, TASK_LANGUAGE_TABLE, DATASET_TO_TASK_LANG
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
    name = f'{task_names}|{lang_names}|{config["method"]}|{config["loss_func"]}|{config["learning_rate"]}'

    matrics_log_path = PROJECT_PATH.joinpath(
        'data',
        'metrics',
        experiment_name,
        name,
        f'{seed}.json',
    )
    if matrics_log_path.exists():
        warnings.warn(
            f"Results for {experiment_name}, f'{task_names}|{lang_names}', with seed {seed} already exist"
        )
        return

    matrics_log_path.parent.mkdir(exist_ok=True, parents=True)

    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', DEFAULT_MLFLOW_TRACKING_URI)
    logger = WandbLogger(
        project='mclt',
    )
    logger.log_hyperparams({**config, 'experiment_tag': experiment_tag})

    checkpoint_path = (
        Path(mlflow_tracking_uri)
        .joinpath(
            'checkpoints',
            logger.experiment.project,
            logger.experiment.name,
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
            EarlyStopping(
                monitor='val_set/f1_score',
                mode='max',
                patience=config['patience'],
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

    datasets = TASK_LANGUAGE_TABLE.loc[
        config['languages'] or slice(None),
        config['tasks'] or slice(None),
    ].dropna()

    if isinstance(datasets, pd.DataFrame):
        datasets = chain.from_iterable(datasets.values)

    datamodule = MultiTaskDataModule(
        [
            {
                'datamodule': DATASETS[dataset](**kwargs),
                'language': DATASET_TO_TASK_LANG[dataset][0],
                'task': DATASET_TO_TASK_LANG[dataset][1],
            }
            for dataset in datasets
        ],
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
    elif isinstance(freeze_backbone, str) and freeze_backbone == 'embeddings':
        transformer.base_model.embeddings.apply(lambda param: param.requires_grad_(False))
    elif isinstance(freeze_backbone, int):
        transformer.base_model.embeddings.apply(lambda param: param.requires_grad_(False))
        for i, layer in enumerate(transformer.base_model.encoder.layer):
            if i >= freeze_backbone:
                break
            layer.apply(lambda param: param.requires_grad_(False))

    return MultitaskTransformerTrainer(
        model=model,
        tasks=tasks,
        learning_rate=config['learning_rate'],
        warmup_steps_ratio=config['warmup_steps_ratio'],
    )
