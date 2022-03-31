from copy import deepcopy

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mclt.datasets import DATASETS
from mclt.datasets.base import BaseDataModule
from mclt.training.trainer import TransformerSentimentTrainer
from mclt.utils.config import load_config
from mclt.utils.experiments import run_experiment
from mclt.utils.seed import set_seeds


def _create_datamodule(config) -> BaseDataModule:
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    datamodule_class = DATASETS[config['dataset']]
    return datamodule_class(
        tokenizer=tokenizer,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        seed=config['data_random_state'],
        max_length=config['max_length'],
        train_sample_size=config['train_sample_size'],
    )


def _create_baseline_model_trainer(config, num_labels):
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

    return TransformerSentimentTrainer(
        model=model,
        learning_rate=config['learning_rate'],
        warmup_steps_ratio=config['warmup_steps_ratio'],
        num_labels=num_labels,
    )


config = load_config('train')
config.update(load_config(config['method']))

datamodule = _create_datamodule(config)
datamodule.prepare_data()
datamodule.setup()

for repeat in range(config['num_repeats']):
    config = deepcopy(config)
    config['model_random_state'] += repeat

    run_experiment(
        config,
        model_trainer=_create_baseline_model_trainer(config, datamodule.num_labels),
        datamodule=datamodule,
        experiment_name=config['dataset'],
        experiment_tag=config['method'],
    )
