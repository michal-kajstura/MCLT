import json
import os
from pathlib import Path
from typing import Any

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor, ModelCheckpoint,
    ModelSummary
)
from pytorch_lightning.loggers import MLFlowLogger

from mclt import DEFAULT_MLFLOW_TRACKING_URI, PROJECT_PATH


def run_experiment(
    config: dict[str, Any],
    model_trainer: LightningModule,
    datamodule: LightningDataModule,
    experiment_name: str,
    experiment_tag: str,
):
    model_trainer._tokenizer = datamodule._tokenizer

    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', DEFAULT_MLFLOW_TRACKING_URI)
    logger = MLFlowLogger(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        tags={'description': experiment_tag},
    )
    logger.log_hyperparams(config)

    checkpoint_path = Path(mlflow_tracking_uri).joinpath(
        'checkpoints',
        logger.experiment_id,
        logger.run_id,
    ).with_suffix('.ckpt')
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    print(checkpoint_path)

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
        precision=config['precision'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        log_every_n_steps=5,
    )

    trainer.fit(
        model=model_trainer,
        datamodule=datamodule,
    )

    metrics, *_ = trainer.test(
        dataloaders=datamodule.val_dataloader(),
    )
    logger.log_metrics({f'test/{k}': v for k, v in metrics.items()})

    model_log_path = PROJECT_PATH.joinpath(
        'data', 'metrics', f'{experiment_name}.json',
    )
    model_log_path.parent.mkdir(exist_ok=True, parents=True)
    with model_log_path.open('w') as file:
        json.dump(metrics, file, indent=2)

    checkpoint_path.unlink(missing_ok=True)
