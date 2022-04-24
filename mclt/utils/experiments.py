import json
import os
import warnings
from pathlib import Path
from typing import Any

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import MLFlowLogger
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from mclt import DEFAULT_MLFLOW_TRACKING_URI, PROJECT_PATH
from mclt.data import DATASETS
from mclt.data.base import MultiTaskDataModule, TaskDefinition
from mclt.modeling.multi_classifier import MultiTaskTransformer
from mclt.training import LOSS_FUNCS
from mclt.training.loss import UncertaintyWeightedLoss
from mclt.training.trainer import MultitaskTransformerTrainer, TransformerTrainer
from mclt.utils.seed import set_seeds


def run_experiment(
    config: dict[str, Any],
    model_trainer: LightningModule,
    datamodule: LightningDataModule,
    experiment_name: str,
    experiment_tag: str,
):
    seed = config['model_random_state']
    matrics_log_path = PROJECT_PATH.joinpath(
        'data',
        'metrics',
        experiment_name,
        f'{seed}.json',
    )
    if matrics_log_path.exists():
        warnings.warn(f"Results for {experiment_name} with seed {seed} already exist")
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
            ModelSummary(max_depth=3),
            LearningRateMonitor('step'),
        ],
        gpus=config['gpus'],
        precision=16 if config['gpus'] else 32,
        accumulate_grad_batches=config['accumulate_grad_batches'],
        log_every_n_steps=5,
        val_check_interval=config['val_check_interval'],
    )

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
    datamodule = MultiTaskDataModule(
        [DATASETS[dataset](**kwargs) for dataset in config['datasets']],
        tokenizer=tokenizer,
        num_workers=config['num_workers'],
        batch_size=config['batch_size'],
    )
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def create_baseline_model_trainer(config, num_labels: int, multilabel: bool):
    set_seeds(config['model_random_state'])

    model = AutoModelForSequenceClassification.from_pretrained(
        config['model'],
        num_labels=num_labels,
    )
    if config['gradient_checkpointing']:
        model.gradient_checkpointing_enable()

    freeze_backbone = config['freeze_backbone']
    if isinstance(freeze_backbone, bool) and freeze_backbone:
        model.base_model.apply(lambda param: param.requires_grad_(False))
    elif isinstance(freeze_backbone, str) and freeze_backbone == 'embeddings':
        model.base_model.embeddings.apply(lambda param: param.requires_grad_(False))

    return TransformerTrainer(
        model=model,
        learning_rate=config['learning_rate'],
        warmup_steps_ratio=config['warmup_steps_ratio'],
        num_labels=num_labels,
        multilabel=multilabel,
    )


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

    return MultitaskTransformerTrainer(
        model=model,
        tasks=tasks,
        learning_rate=config['learning_rate'],
        warmup_steps_ratio=config['warmup_steps_ratio'],
    )
