import json
import logging
import os
import warnings
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict

import datasets
import pandas as pd
import transformers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from requests.exceptions import ConnectionError
from retry import retry
from transformers import AutoModel, AutoTokenizer, RobertaModel

from mclt import (
    DEFAULT_MLFLOW_TRACKING_URI,
    PROJECT_PATH,
)
from mclt.data import DATASETS, TASK_LANGUAGE_TABLE, DATASET_TO_TASK_LANG
from mclt.data.base import MultiTaskDataModule, TaskDefinition
from mclt.modeling.adapter_roberta import add_adapter_layers
from mclt.modeling.multi_classifier import MultiTaskTransformer
from mclt.training import LOSS_FUNCS, GradSurgeryLoss
from mclt.training.trainer import MultitaskTransformerTrainer, GradSurgeryTrainer
from mclt.utils.seed import set_seeds

transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@retry(
    exceptions=(ConnectionError, ValueError),
    tries=30,
    delay=30,
    logger=logger,
)
def run_experiment(
    config: dict[str, Any],
    create_datamodule: Callable[[Dict], MultiTaskDataModule],
    create_model_trainer: Callable[[Dict, dict[str, TaskDefinition]], MultitaskTransformerTrainer],
    experiment_name: str,
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

    datamodule = create_datamodule(config)
    tasks = datamodule.tasks
    model_trainer = create_model_trainer(config, tasks)

    matrics_log_path.parent.mkdir(exist_ok=True, parents=True)
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', DEFAULT_MLFLOW_TRACKING_URI)
    from pytorch_lightning.loggers import WandbLogger
    logger = WandbLogger(
        project='MTML',
    )
    logger.log_hyperparams(
        {
            **config,
            'experiment_name': experiment_name,
        }
    )

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
        max_steps=config['max_steps'] * len(tasks),
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
        val_check_interval=config['val_check_interval'] * len(tasks),
        num_sanity_val_steps=0,
    )

    # logger.experiment.watch(model_trainer, log='all')
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
    logger.experiment.finish()


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
        add_pooling_layer=False,
    )

    loss_func = LOSS_FUNCS[config['loss_func']](tasks)

    model = MultiTaskTransformer(
        transformer=transformer,
        tasks=tasks,
        loss_func=loss_func,
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

    if config['method'] == 'adapter':
        add_adapter_layers_and_freeze(transformer)

    trainer_class = GradSurgeryTrainer if isinstance(loss_func, GradSurgeryLoss) else MultitaskTransformerTrainer
    return trainer_class(
        model=model,
        tasks=tasks,
        learning_rate=config['learning_rate'],
        warmup_steps_ratio=config['warmup_steps_ratio'],
    )


def add_adapter_layers_and_freeze(transformer):
    add_adapter_layers(transformer.base_model)
    transformer.base_model.apply(lambda param: param.requires_grad_(False))
    for layer in transformer.base_model.encoder.layer:
        layer.output.adapter.requires_grad_(True)


def run_adaptune_experiment(
    config: dict[str, Any],
    create_datamodule: Callable[[Dict], MultiTaskDataModule],
    create_model_trainer: Callable[[Dict, dict[str, TaskDefinition]], MultitaskTransformerTrainer],
    experiment_name: str,
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

    datamodule = create_datamodule(config)
    tasks = datamodule.tasks
    model_trainer = create_model_trainer(config, tasks)

    matrics_log_path.parent.mkdir(exist_ok=True, parents=True)
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', DEFAULT_MLFLOW_TRACKING_URI)
    from pytorch_lightning.loggers import WandbLogger
    logger = WandbLogger(
        project='MTML',
    )
    logger.log_hyperparams(
        {
            **config,
            'experiment_name': experiment_name,
        }
    )

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

    def _create_trainer(max_steps, val_check_interval):
        return Trainer(
            max_steps=max_steps,
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
            val_check_interval=val_check_interval,
            num_sanity_val_steps=0,
        )

    ###############################
    # shared training
    adapter_finetune_steps_ratio = config['adapter_finetune_steps_ratio']
    shared_training_steps_ratio = 1.0 - adapter_finetune_steps_ratio
    per_task_finetune_ratio = adapter_finetune_steps_ratio / len(tasks)

    trainer = _create_trainer(
        int(config['max_steps'] * len(tasks) * shared_training_steps_ratio),
        int(config['val_check_interval'] * len(tasks) * shared_training_steps_ratio),
    )
    trainer.fit(
        model=model_trainer,
        datamodule=datamodule,
    )

    model_trainer.cpu()
    checkpoint_callback = next(c for c in trainer.callbacks if isinstance(c, ModelCheckpoint))
    best_model_path = checkpoint_callback.best_model_path
    model_trainer.load_from_checkpoint(
        best_model_path,
        model=model_trainer.model,
        tasks=model_trainer.tasks,
    )
    shared_state_dict = model_trainer.state_dict()

    print('Finished shared training')

    ###############################
    # individual tasks training

    metrics = {}
    f1_scores = []
    for task_name, task in tasks.items():
        print(f'training {task_name} adapter')
        datamodule.set_datasets([task_name])

        trainer = _create_trainer(
            int(config['max_steps'] * per_task_finetune_ratio),
            int(config['val_check_interval'] * per_task_finetune_ratio),
        )

        config['learning_rate'] = config['adapter_learning_rate']
        model_trainer = create_model_trainer(config, {task_name: task})
        keys = model_trainer.load_state_dict(shared_state_dict, strict=False)
        other_tasks = set(tasks).difference({task_name})
        assert all(key.split('.')[-3] in other_tasks for key in keys.unexpected_keys)

        transformer: RobertaModel = model_trainer.model.transformer
        add_adapter_layers_and_freeze(transformer)
        trainer.fit(
            model=model_trainer,
            datamodule=datamodule,
        )

        task_metrics, *_ = trainer.test(
            dataloaders=datamodule.val_dataloader(),
        )
        metrics.update(task_metrics)
        f1_scores.append(task_metrics['test_set/f1_score'])

    metrics['test_set/f1_score'] = sum(f1_scores) / len(f1_scores)
    logger.log_metrics({f'test/{k}': v for k, v in metrics.items()})

    with matrics_log_path.open('w') as file:
        json.dump(metrics, file, indent=2)

    for ckpt_path in checkpoint_path.parent.iterdir():
        ckpt_path.unlink(missing_ok=True)
    logger.experiment.finish()
